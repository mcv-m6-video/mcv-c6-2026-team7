"""
UNet variant with Squeeze-and-Excitation (SE) blocks after every encoder,
bottleneck, and decoder convolution. SE blocks recalibrate channel-wise
feature responses at near-zero parameter cost.
"""

import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

from model.modules import BaseRGBModel, FCLayers, step


class SEBlock1d(nn.Module):
    """Channel-wise Squeeze-and-Excitation for 1-D feature maps (B, C, T)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        s = self.gap(x).view(b, c)
        s = self.fc(s).view(b, c, 1)
        return x * s


class ConvSE(nn.Module):
    """Conv1d + ReLU + SE block as a single reusable unit."""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, reduction=16):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.se = SEBlock1d(out_ch, reduction=reduction)

    def forward(self, x):
        return self.se(F.relu(self.conv(x)))


class TemporalUNetSE(nn.Module):
    def __init__(self, channels, hidden_channels=256, out_channels=128, reduction=16):
        super().__init__()
        # Encoder
        self.enc1 = ConvSE(channels, hidden_channels, reduction=reduction)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = ConvSE(hidden_channels, hidden_channels * 2, reduction=reduction)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = ConvSE(hidden_channels * 2, hidden_channels * 4, reduction=reduction)
        self.pool3 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = ConvSE(hidden_channels * 4, hidden_channels * 4,
                                 reduction=reduction)

        # Decoder
        self.dec3 = ConvSE(hidden_channels * 8, hidden_channels * 2, reduction=reduction)
        self.dec2 = ConvSE(hidden_channels * 4, hidden_channels, reduction=reduction)
        self.dec1 = ConvSE(hidden_channels * 2, out_channels, reduction=reduction)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = F.interpolate(b, size=e3.shape[-1], mode='linear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[-1], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[-1], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return d1


class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            self._feature_arch = args.feature_arch

            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()
                self._d = feat_dim
            else:
                raise NotImplementedError(args._feature_arch)

            self._features = features

            self.unet_hidden = 256
            self.unet_out_dim = self.unet_hidden // 2
            self._temporal_unet = TemporalUNetSE(
                channels=self._d,
                hidden_channels=self.unet_hidden,
                out_channels=self.unet_out_dim,
                reduction=16,
            )

            self._fc = FCLayers(self.unet_out_dim, args.num_classes + 1)

            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])

            self.standarization = T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        def forward(self, x):
            x = self.normalize(x)
            batch_size, clip_len, channels, height, width = x.shape

            if self.training:
                x = self.augment(x)

            x = self.standarize(x)

            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._d)

            im_feat = im_feat.permute(0, 2, 1)
            im_feat = self._temporal_unet(im_feat)
            im_feat = im_feat.permute(0, 2, 1)

            return self._fc(im_feat)

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
            print('Model params:', sum(p.numel() for p in self.parameters()))

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
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

        weights = torch.tensor([1.0] + [5.0] * self._num_classes,
                               dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label'].to(self.device).long()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1)
                    label = label.view(-1)
                    loss = F.cross_entropy(pred, label, reduction='mean', weight=weights)

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
            return torch.softmax(pred, dim=-1).cpu().numpy()
