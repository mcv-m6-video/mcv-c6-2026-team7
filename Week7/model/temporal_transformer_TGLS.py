"""
File containing the temporal transformer model.
"""

#Standard imports
import torch
from torch import nn
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F


#Local imports
from model.modules import BaseRGBModel, FCLayers, create_backbone, step


class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

            self._features, self._d = create_backbone(args)

            # Temporal modelling via Transformer encoder
            self._pos_embed = nn.Parameter(torch.randn(1, args.clip_len, self._d) * 0.02)
            transformer_layers = int(getattr(args, 'transformer_layers', 3))
            transformer_dropout = float(getattr(args, 'transformer_dropout', 0.25))
            transformer_nhead = int(getattr(args, 'transformer_nhead', 8))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self._d,
                nhead=transformer_nhead,
                dim_feedforward=self._d * 2,
                dropout=transformer_dropout,
                batch_first=True,
                norm_first=True,
            )
            self._temporal = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

            # MLP for classification
            self._fc = FCLayers(self._d, args.num_classes+1) # +1 for background class (we now perform per-frame classification with softmax, therefore we have the extra background class)

            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len = x.shape[0], x.shape[1] #B, T

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats

            im_feat = self._features(x) #B, T, D

            # Temporal modelling
            im_feat = im_feat + self._pos_embed
            im_feat = self._temporal(im_feat) #B, T, D

            #MLP
            im_feat = self._fc(im_feat) #B, T, num_classes+1

            return im_feat

        def normalize(self, x):
            return x / 255.

        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self):
            print('Model params:',
                sum(p.numel() for p in self.parameters()))

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and getattr(args, 'device', 'cpu') == "cuda":
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        # --- GAUSSIAN KERNEL FOR TEMPORAL SMOOTHING ---
        window_size = int(getattr(self._args, 'label_smoothing_window', 5))
        sigma = float(getattr(self._args, 'label_smoothing_sigma', 0.55))
        C = self._num_classes

        # 1D Gaussian kernel
        x_coords = torch.arange(-window_size // 2 + 1., window_size // 2 + 1., device=self.device)
        gauss = torch.exp(-x_coords**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()

        # Reshape for depthwise 1D convolution
        kernel = gauss.view(1, 1, window_size).repeat(C, 1, 1)
        padding = window_size // 2
        # ----------------------------------------------

        weights = torch.tensor([1.0] + [5.0] * (self._num_classes), dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).long()

                # ------ TEMPORAL LABEL SMOOTHING ------
                # 1. Convert integer labels to one-hot: (B, T, C+1)
                y_one_hot = F.one_hot(label, num_classes=C + 1).float()

                # 2. Extract active classes and permute for conv1d: (B, C, T)
                y_active = y_one_hot[:, :, 1:].permute(0, 2, 1)

                # 3. Apply depthwise 1D convolution to smooth active classes temporally
                y_active_smooth = F.conv1d(y_active, kernel, padding=padding, groups=C)

                # Prevent numerical overflow in edge cases (e.g., overlapping events)
                y_active_smooth = torch.clamp(y_active_smooth, min=0.0, max=1.0)

                # 4. Recompute background probability (1.0 minus the sum of active probabilities)
                y_bg = 1.0 - y_active_smooth.sum(dim=1, keepdim=True)
                y_bg = torch.clamp(y_bg, min=0.0, max=1.0)

                # 5. Concatenate background and smoothed active classes back together: (B, C+1, T)
                y_smooth = torch.cat([y_bg, y_active_smooth], dim=1)

                # 6. Permute back to (B, T, C+1) and flatten into (B*T, C+1) for the loss function
                y_smooth = y_smooth.permute(0, 2, 1).reshape(-1, C + 1)
                # --------------------------------------

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1) # B*T, num_classes + 1

                    # Compute cross entropy using the soft labels
                    # PyTorch handles target probabilities automatically if shapes match
                    loss = F.cross_entropy(
                            pred, y_smooth, reduction='mean', weight=weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            # apply sigmoid
            pred = torch.softmax(pred, dim=-1)

            return pred.cpu().numpy()
