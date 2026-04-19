"""
File containing the non-residual/configurable BiGRU model.
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


class TemporalBiGRU(nn.Module):
    """
    Bidirectional GRU block with optional residual connection.
    """

    def __init__(self, dim, num_layers=2, dropout=0.3, use_residual=False):
        super().__init__()

        if num_layers < 1:
            raise ValueError('gru_num_layers must be >= 1')
        if dim % 2 != 0:
            raise ValueError('Feature dimension must be even for bidirectional GRU with hidden_size=dim//2')

        self.use_residual = bool(use_residual)
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(dim) if self.use_residual else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, T, D)
        out, _ = self.gru(x)
        out = self.dropout(out)
        if self.use_residual:
            out = x + out
        return self.norm(out)


class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            if args is None:
                raise ValueError('Model.Impl requires args')

            self._feature_arch = args.feature_arch
            self._features, self._d = create_backbone(args)

            gru_num_layers = int(getattr(args, 'gru_num_layers', 2))
            gru_dropout = float(getattr(args, 'gru_dropout', 0.3))
            # Default False in this module, but can be enabled from config.
            gru_residual = bool(getattr(args, 'gru_residual', False))

            self._temporal = TemporalBiGRU(
                dim=self._d,
                num_layers=gru_num_layers,
                dropout=gru_dropout,
                use_residual=gru_residual,
            )

            # MLP for classification
            self._fc = FCLayers(self._d, args.num_classes + 1)  # +1 for background class

            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        def forward(self, x):
            x = self.normalize(x)  # Normalize to 0-1

            if self.training:
                x = self.augment(x)  # augmentation per-batch

            x = self.standarize(x)  # standarization imagenet stats

            # Backbones adapter returns B, T, D
            im_feat = self._features(x)

            # Temporal modelling
            im_feat = self._temporal(im_feat)

            #MLP
            im_feat = self._fc(im_feat)

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
        if args is None:
            raise ValueError('Model requires args')

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
        gauss = torch.exp(-x_coords ** 2 / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()

        # Reshape for depthwise 1D convolution
        kernel = gauss.view(1, 1, window_size).repeat(C, 1, 1)
        padding = window_size // 2
        # ----------------------------------------------

        weights = torch.tensor([1.0] + [5.0] * (self._num_classes), dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch in tqdm(loader):
                frame = batch['frame'].to(self.device).float()
                label = batch['label'].to(self.device).long()

                # ------ TEMPORAL LABEL SMOOTHING ------
                y_one_hot = F.one_hot(label, num_classes=C + 1).float()
                y_active = y_one_hot[:, :, 1:].permute(0, 2, 1)
                y_active_smooth = F.conv1d(y_active, kernel, padding=padding, groups=C)
                y_active_smooth = torch.clamp(y_active_smooth, min=0.0, max=1.0)

                y_bg = 1.0 - y_active_smooth.sum(dim=1, keepdim=True)
                y_bg = torch.clamp(y_bg, min=0.0, max=1.0)

                y_smooth = torch.cat([y_bg, y_active_smooth], dim=1)
                y_smooth = y_smooth.permute(0, 2, 1).reshape(-1, C + 1)
                # --------------------------------------

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1)
                    loss = F.cross_entropy(
                        pred, y_smooth, reduction='mean', weight=weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            pred = torch.softmax(pred, dim=-1)

            return pred.cpu().numpy()
