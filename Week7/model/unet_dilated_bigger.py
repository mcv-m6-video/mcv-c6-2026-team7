"""
Larger dilated UNet variant targeting ~13-14 M total parameters.
Compared to unet_dilated.py:
  - hidden_channels 128 → 256  (encoder: 256 → 512 → 1024)
  - DilatedBottleneck reduction 4 → 3  (mid = channels//3 ≈ 341 for 1024 ch)
    bottleneck goes from ~1.6 M to ~2.5 M, matching the baseline param budget.
"""

import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

from model.modules import BaseRGBModel, FCLayers, step


class DilatedBottleneck(nn.Module):
    """Lightweight multi-scale dilated bottleneck.
    Projects channels down by `reduction` before the parallel dilated branches,
    then projects back — same multi-scale receptive field at a fraction of the cost.
    """

    def __init__(self, channels, reduction=3):
        super().__init__()
        mid = max(channels // reduction, 32)
        self.down = nn.Conv1d(channels, mid, kernel_size=1)
        self.d1 = nn.Conv1d(mid, mid, kernel_size=3, padding=1, dilation=1)
        self.d2 = nn.Conv1d(mid, mid, kernel_size=3, padding=2, dilation=2)
        self.d4 = nn.Conv1d(mid, mid, kernel_size=3, padding=4, dilation=4)
        self.up = nn.Conv1d(mid * 3, channels, kernel_size=1)

    def forward(self, x):
        x_down = F.relu(self.down(x))
        return F.relu(self.up(torch.cat([
            F.relu(self.d1(x_down)),
            F.relu(self.d2(x_down)),
            F.relu(self.d4(x_down)),
        ], dim=1)))


class TemporalUNetDilated(nn.Module):
    def __init__(self, channels, hidden_channels=256, out_channels=128):
        super().__init__()
        # Encoder — dilation grows with depth to widen receptive field
        self.enc1 = nn.Conv1d(channels, hidden_channels,
                              kernel_size=3, padding=1, dilation=1)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = nn.Conv1d(hidden_channels, hidden_channels * 2,
                              kernel_size=3, padding=2, dilation=2)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = nn.Conv1d(hidden_channels * 2, hidden_channels * 4,
                              kernel_size=3, padding=4, dilation=4)
        self.pool3 = nn.MaxPool1d(2)

        # Multi-scale dilated bottleneck (reduction=3 → mid ≈ channels//3)
        self.bottleneck = DilatedBottleneck(hidden_channels * 4, reduction=3)

        # Decoder — standard convolutions (skip connections already provide context)
        self.dec3 = nn.Conv1d(hidden_channels * 8, hidden_channels * 2,
                              kernel_size=3, padding=1)
        self.dec2 = nn.Conv1d(hidden_channels * 4, hidden_channels,
                              kernel_size=3, padding=1)
        self.dec1 = nn.Conv1d(hidden_channels * 2, out_channels,
                              kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = F.relu(self.enc1(x))
        p1 = self.pool1(e1)

        e2 = F.relu(self.enc2(p1))
        p2 = self.pool2(e2)

        e3 = F.relu(self.enc3(p2))
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = F.interpolate(b, size=e3.shape[-1], mode='linear', align_corners=False)
        d3 = F.relu(self.dec3(torch.cat([d3, e3], dim=1)))

        d2 = F.interpolate(d3, size=e2.shape[-1], mode='linear', align_corners=False)
        d2 = F.relu(self.dec2(torch.cat([d2, e2], dim=1)))

        d1 = F.interpolate(d2, size=e1.shape[-1], mode='linear', align_corners=False)
        d1 = F.relu(self.dec1(torch.cat([d1, e1], dim=1)))

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
            self.unet_out_dim = 128
            self._temporal_unet = TemporalUNetDilated(
                channels=self._d,
                hidden_channels=self.unet_hidden,
                out_channels=self.unet_out_dim,
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
