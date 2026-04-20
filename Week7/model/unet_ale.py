"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

#Local imports
from model.modules import BaseRGBModel, FCLayers, step

class TemporalUNet(nn.Module):
    def __init__(self, channels, hidden_channels=256, out_channels=128, dropout_p=0.0):
        super().__init__()
        
        # Dropout layer
        self.dropout = nn.Dropout1d(p=dropout_p)
        
        # Encoder
        self.enc1 = nn.Conv1d(channels, hidden_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = nn.Conv1d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1)

        # Decoder
        self.dec3 = nn.Conv1d(hidden_channels * 8, hidden_channels * 2, kernel_size=3, padding=1)
        self.dec2 = nn.Conv1d(hidden_channels * 4, hidden_channels, kernel_size=3, padding=1)
        self.dec1 = nn.Conv1d(hidden_channels * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = F.relu(self.enc1(x))
        p1 = self.pool1(e1)
        p1 = self.dropout(p1)

        e2 = F.relu(self.enc2(p1))
        p2 = self.pool2(e2)
        p2 = self.dropout(p2)

        e3 = F.relu(self.enc3(p2))
        p3 = self.pool3(e3)
        p3 = self.dropout(p3)

        # Bottleneck
        b = F.relu(self.bottleneck(p3))
        b = self.dropout(b)

        # Decoder
        d3 = F.interpolate(b, size=e3.shape[-1], mode='linear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1) 
        d3 = F.relu(self.dec3(d3))
        d3 = self.dropout(d3)

        d2 = F.interpolate(d3, size=e2.shape[-1], mode='linear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1) 
        d2 = F.relu(self.dec2(d2))
        d2 = self.dropout(d2)

        d1 = F.interpolate(d2, size=e1.shape[-1], mode='linear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1) 
        d1 = F.relu(self.dec1(d1))
        d1 = self.dropout(d1)

        return d1

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch
            self.unet_dropout = args.unet_dropout

            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features

                # Remove final classification layer
                features.head.fc = nn.Identity()
                self._d = feat_dim

            else:
                raise NotImplementedError(args._feature_arch)

            self._features = features

            # Temporal U-Net
            self.unet_hidden = 256
            self.unet_out_dim = self.unet_hidden // 2
            self._temporal_unet = TemporalUNet(
                channels=self._d,
                hidden_channels=self.unet_hidden,
                out_channels=self.unet_out_dim,
                dropout_p=self.unet_dropout
            )

            # MLP for classification
            self._fc = FCLayers(self.unet_out_dim, args.num_classes+1) # +1 for background class (we now perform per-frame classification with softmax, therefore we have the extra background class)

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
            batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats
                        
            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._d) #B, T, D

            # Temporal Processing
            im_feat = im_feat.permute(0, 2, 1) # Permute to (B, D, T) for Conv1D
            im_feat = self._temporal_unet(im_feat)
            im_feat = im_feat.permute(0, 2, 1) # Permute back to (B, T, D) for the FC layer

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
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args
        self._model.to(self.device)
        self._num_classes = args.num_classes
        self.label_smoothing = args.label_smoothing
        self.label_smo_window = args.label_smo_window

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()
        
        # --- TEMPORAL LABEL SMOOTHING KERNEL ---
        window_size = self.label_smo_window
        C = self._num_classes
        
        half_w = window_size // 2
        x_coords = torch.arange(-half_w, half_w + 1., device=self.device)
        
        if self.label_smoothing == 'gaussian':
            sigma = 0.55
            # Gaussian peak: [0.135, 0.606, 1.0, 0.606, 0.135]
            base_kernel = torch.exp(-x_coords**2 / (2 * sigma**2))
            
        elif self.label_smoothing == 'triangle':
            # Triangular peak: [0.33, 0.67, 1.0, 0.67, 0.33]
            base_kernel = 1.0 - (torch.abs(x_coords) / (half_w + 1))
            
        elif self.label_smoothing == 'rectangle':
            # Uniform weight: [1.0, 1.0, 1.0, 1.0, 1.0]
            base_kernel = torch.ones_like(x_coords)
        
        elif self.label_smoothing == 'none':
            # Delta function: [0.0, 0.0, 1.0, 0.0, 0.0]
            base_kernel = torch.zeros_like(x_coords)
            base_kernel[half_w] = 1.0
            
        else:
            raise ValueError(f"Unsupported smoothing type: {self.label_smoothing}")
            
        # Reshape for depthwise 1D convolution
        kernel = base_kernel.view(1, 1, window_size).repeat(C, 1, 1)
        padding = half_w
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
                
                # Prevent numerical instability
                y_active_smooth = torch.clamp(y_active_smooth, min=0.0, max=1.0)

                # 4. Recompute background probability 
                y_bg = 1.0 - y_active_smooth.sum(dim=1, keepdim=True)
                y_bg = torch.clamp(y_bg, min=0.0) # Just clamp the bottom, no need for max=1.0

                # 5. Concatenate
                y_smooth = torch.cat([y_bg, y_active_smooth], dim=1)

                # 6. Frame-wise normalization to handle overlapping action curves
                y_smooth = y_smooth / y_smooth.sum(dim=1, keepdim=True)

                # 7. Permute back
                y_smooth = y_smooth.permute(0, 2, 1).reshape(-1, C + 1)
                # --------------------------------------

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1) # B*T, num_classes
                    label = label.view(-1) # B*T
                    loss = F.cross_entropy(
                            pred, label, reduction='mean', weight = weights)

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