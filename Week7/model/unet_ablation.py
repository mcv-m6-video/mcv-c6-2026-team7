"""
File containing the main model.

Combines:
  - SE blocks (SEBlock1d, ConvSE)
  - Multi-scale dilated bottleneck (DilatedBottleneck)
  - Temporal U-Net with dropout (TemporalUNet)
  - Pluggable 1-D pooling via model.pooling (max / avg / lp / concat /
    stochastic / blurpool)
  - Temporal label smoothing (gaussian / triangle / rectangle / none)
"""

# Standard imports
import torch
from torch import nn
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

# Local imports
from model.modules import BaseRGBModel, FCLayers, create_backbone, step
from model.pooling import make_pool, ConcatPool1d, POOL_TYPES


# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────────────────────

class SEBlock1d(nn.Module):
    """Channel-wise Squeeze-and-Excitation for 1-D feature maps (B, C, T)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        s = self.gap(x).view(b, c)
        s = self.fc(s).view(b, c, 1)
        return x * s


class ConvSE(nn.Module):
    """Conv1d + ReLU + SE block — activation is baked in."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 padding: int = 1, dilation: int = 1, reduction: int = 16):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                              padding=padding, dilation=dilation)
        self.se   = SEBlock1d(out_ch, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(F.relu(self.conv(x)))


class DilatedBottleneck(nn.Module):
    """
    Multi-scale dilated bottleneck: projects down, applies parallel dilated
    convs at rates 1 / 2 / 4, then projects back up.
    Activation is baked in (returns ReLU output).
    """

    def __init__(self, channels: int, reduction: int = 3):
        super().__init__()
        mid       = max(channels // reduction, 32)
        self.down = nn.Conv1d(channels, mid, kernel_size=1)
        self.d1   = nn.Conv1d(mid, mid, kernel_size=3, padding=1,  dilation=1)
        self.d2   = nn.Conv1d(mid, mid, kernel_size=3, padding=2,  dilation=2)
        self.d4   = nn.Conv1d(mid, mid, kernel_size=3, padding=4,  dilation=4)
        self.up   = nn.Conv1d(mid * 3, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_down = F.relu(self.down(x))
        return F.relu(self.up(torch.cat([
            F.relu(self.d1(x_down)),
            F.relu(self.d2(x_down)),
            F.relu(self.d4(x_down)),
        ], dim=1)))


# ──────────────────────────────────────────────────────────────────────────────
# Temporal U-Net
# ──────────────────────────────────────────────────────────────────────────────

class TemporalUNet(nn.Module):
    """
    1-D Temporal U-Net with:
      • Pluggable pooling  (max / avg / lp / concat / stochastic / blurpool)
      • Optional Squeeze-and-Excitation on every encoder/decoder conv
      • Optional multi-scale dilated bottleneck
      • Per-layer Dropout1d

    Channel bookkeeping for ConcatPool
    ------------------------------------
    ConcatPool doubles the channel count after every pooling step.  To keep
    the decoder sizes consistent we pre-compensate by halving the *base*
    encoder width (h = hidden_channels // 2) so that after concatenation the
    tensors entering enc2/enc3 match what the decoder expects — identical
    behaviour to the standalone pooling model in file 2.

    Activation convention
    ----------------------
    ConvSE and DilatedBottleneck already apply ReLU internally; plain Conv1d
    does not.  The `act` lambda in forward() handles this transparently.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 256,
        out_channels: int = 128,
        dropout_p: float = 0.0,
        use_dilated: bool = False,
        use_se: bool = False,
        se_reduction: int = 16,
        pool_type: str = "max",
        use_temporal_bottleneck: bool = True,
    ):
        super().__init__()

        if pool_type not in POOL_TYPES:
            raise ValueError(
                f"pool_type must be one of {POOL_TYPES}, got '{pool_type}'"
            )

        self.use_dilated = use_dilated
        self.use_se      = use_se
        self.pool_type   = pool_type
        self.use_temporal_bottleneck = use_temporal_bottleneck
        self.dropout     = nn.Dropout1d(p=dropout_p)

        # ConcatPool doubles channels only when temporal downsampling is active.
        is_concat = (pool_type == "concat") and use_temporal_bottleneck
        h = hidden_channels // 2 if is_concat else hidden_channels

        # Channel multiplier introduced by each pooling step
        m = 2 if is_concat else 1

        # ── Encoder conv factory ─────────────────────────────────────────
        def enc_conv(in_ch: int, out_ch: int, dilation: int = 1) -> nn.Module:
            pad = dilation  # kernel_size=3 → padding=dilation preserves length
            if use_se:
                return ConvSE(in_ch, out_ch, kernel_size=3, padding=pad,
                              dilation=dilation, reduction=se_reduction)
            return nn.Conv1d(in_ch, out_ch, kernel_size=3,
                             padding=pad, dilation=dilation)

        def dec_conv(in_ch: int, out_ch: int) -> nn.Module:
            if use_se:
                return ConvSE(in_ch, out_ch, kernel_size=3, padding=1,
                              reduction=se_reduction)
            return nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        # ── Encoder ──────────────────────────────────────────────────────
        # Dilation grows with depth when use_dilated=True (encoder path only;
        # the bottleneck handles multi-scale when use_dilated=True).
        self.enc1 = enc_conv(channels,    h,     dilation=1)
        self.enc2 = enc_conv(h * m,       h * 2, dilation=2 if use_dilated else 1)
        self.enc3 = enc_conv(h * 2 * m,   h * 4, dilation=4 if use_dilated else 1)

        if use_temporal_bottleneck:
            # Pools act on the output of the preceding encoder conv.
            # BlurPool needs the channel count at that point.
            self.pool1 = make_pool(pool_type, channels=h,     kernel_size=2)
            self.pool2 = make_pool(pool_type, channels=h * 2, kernel_size=2)
            self.pool3 = make_pool(pool_type, channels=h * 4, kernel_size=2)
        else:
            # Keep temporal resolution fixed across the full temporal U-Net.
            self.pool1 = nn.Identity()
            self.pool2 = nn.Identity()
            self.pool3 = nn.Identity()

        # ── Bottleneck ───────────────────────────────────────────────────
        btn_in = h * 4 * m   # = pool3 output channels

        if use_dilated:
            # DilatedBottleneck expects & produces `btn_in` channels
            self.bottleneck    = DilatedBottleneck(btn_in, reduction=3)
            # Optional SE gate after the dilated block
            self.bottleneck_se = (SEBlock1d(btn_in, reduction=se_reduction)
                                  if use_se else nn.Identity())
        elif use_se:
            self.bottleneck    = ConvSE(btn_in, btn_in, reduction=se_reduction)
            self.bottleneck_se = nn.Identity()
        else:
            self.bottleneck    = nn.Conv1d(btn_in, btn_in, kernel_size=3, padding=1)
            self.bottleneck_se = nn.Identity()

        btn_out = btn_in   # bottleneck is channel-preserving

        # ── Decoder ──────────────────────────────────────────────────────
        # Skip connections come from enc outputs (before pooling) → always h*N
        e3_ch, e2_ch, e1_ch = h * 4, h * 2, h

        self.dec3 = dec_conv(btn_out + e3_ch, h * 2)
        self.dec2 = dec_conv(h * 2   + e2_ch, h)
        self.dec1 = dec_conv(h       + e1_ch, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ConvSE / DilatedBottleneck activate internally; plain Conv1d does not.
        act = (lambda t: t) if self.use_se else F.relu

        # ── Encoder ──────────────────────────────────────────────────────
        e1 = act(self.enc1(x))          # (B, h,       T)
        p1 = self.dropout(self.pool1(e1))   # (B, h[*2],   T/2)

        e2 = act(self.enc2(p1))         # (B, h*2,     T/2)
        p2 = self.dropout(self.pool2(e2))

        e3 = act(self.enc3(p2))         # (B, h*4,     T/4)
        p3 = self.dropout(self.pool3(e3))

        # ── Bottleneck ───────────────────────────────────────────────────
        if self.use_dilated:
            # DilatedBottleneck activates internally; bottleneck_se is SE or Identity
            b = self.bottleneck_se(self.bottleneck(p3))
        else:
            b = act(self.bottleneck(p3))
        b = self.dropout(b)

        # ── Decoder (skip connections from *pre-pool* encoder outputs) ───
        d3 = F.interpolate(b,  size=e3.shape[-1], mode='linear', align_corners=False)
        d3 = act(self.dec3(torch.cat([d3, e3], dim=1)))
        d3 = self.dropout(d3)

        d2 = F.interpolate(d3, size=e2.shape[-1], mode='linear', align_corners=False)
        d2 = act(self.dec2(torch.cat([d2, e2], dim=1)))
        d2 = self.dropout(d2)

        d1 = F.interpolate(d2, size=e1.shape[-1], mode='linear', align_corners=False)
        d1 = act(self.dec1(torch.cat([d1, e1], dim=1)))
        d1 = self.dropout(d1)

        return d1   # (B, out_channels, T)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level Model
# ──────────────────────────────────────────────────────────────────────────────

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            self._feature_arch = args.feature_arch
            self.unet_dropout  = args.unet_dropout

            self._features, self._d = create_backbone(args)

            # ── Temporal U-Net ────────────────────────────────────────────
            self.unet_hidden  = 256
            self.unet_out_dim = self.unet_hidden // 2

            # pool_type is optional; falls back to 'max' (original behaviour)
            pool_type = getattr(args, 'pool_type', 'max')

            self._temporal_unet = TemporalUNet(
                channels=self._d,
                hidden_channels=self.unet_hidden,
                out_channels=self.unet_out_dim,
                dropout_p=self.unet_dropout,
                use_dilated=getattr(args, 'use_dilated', False),
                use_se=getattr(args, 'use_se', False),
                se_reduction=getattr(args, 'se_reduction', 16),
                pool_type=pool_type,
                use_temporal_bottleneck=getattr(args, 'use_temporal_bottleneck', True),
            )

            # ── Classification head ───────────────────────────────────────
            # +1 for the background class
            self._fc = FCLayers(self.unet_out_dim, args.num_classes + 1)

            # ── Augmentations ─────────────────────────────────────────────
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)],                   p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))],     p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))],     p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))],       p=0.25),
                T.RandomApply([T.GaussianBlur(5)],                        p=0.25),
                T.RandomHorizontalFlip(),
            ])

            # ── Standardisation ───────────────────────────────────────────
            self.standarization = T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
            ])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.normalize(x)                                   # → [0, 1]

            if self.training:
                x = self.augment(x)

            x = self.standarize(x)

            im_feat = self._features(x)                              # (B, T, D)

            # Temporal processing
            im_feat = im_feat.permute(0, 2, 1)                      # (B, D, T)
            im_feat = self._temporal_unet(im_feat)
            im_feat = im_feat.permute(0, 2, 1)                      # (B, T, D)

            # Classification
            im_feat = self._fc(im_feat)                             # (B, T, C+1)
            return im_feat

        def normalize(self, x: torch.Tensor) -> torch.Tensor:
            return x / 255.

        def augment(self, x: torch.Tensor) -> torch.Tensor:
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def standarize(self, x: torch.Tensor) -> torch.Tensor:
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self):
            print('Model params:',
                  sum(p.numel() for p in self.parameters()))

    # ── Outer Model shell ──────────────────────────────────────────────────────

    def __init__(self, args=None):
        self.device = "cpu"
        if (torch.cuda.is_available()
                and "device" in args
                and args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args        = args
        self._model.to(self.device)
        self._num_classes = args.num_classes
        self.label_smoothing  = args.label_smoothing
        self.label_smo_window = args.label_smo_window
        self.LS_gaussian_sigma = args.LS_gaussian_sigma

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        # ── Temporal label-smoothing kernel ──────────────────────────────
        window_size = self.label_smo_window
        C           = self._num_classes
        half_w      = window_size // 2

        x_coords = torch.arange(-half_w, half_w + 1., device=self.device)

        if self.label_smoothing == 'gaussian':
            sigma       = self.LS_gaussian_sigma
            base_kernel = torch.exp(-x_coords ** 2 / (2 * sigma ** 2))

        elif self.label_smoothing == 'triangle':
            base_kernel = 1.0 - (torch.abs(x_coords) / (half_w + 1))

        elif self.label_smoothing == 'rectangle':
            base_kernel = torch.ones_like(x_coords)

        elif self.label_smoothing == 'none':
            base_kernel = torch.zeros_like(x_coords)
            base_kernel[half_w] = 1.0

        else:
            raise ValueError(
                f"Unsupported smoothing type: {self.label_smoothing}"
            )

        # Reshape for depthwise 1-D convolution: (C, 1, window_size)
        kernel  = base_kernel.view(1, 1, window_size).repeat(C, 1, 1)
        padding = half_w
        # ─────────────────────────────────────────────────────────────────

        weights = torch.tensor(
            [1.0] + [5.0] * C, dtype=torch.float32
        ).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label'].to(self.device).long()

                # ── Temporal label smoothing ──────────────────────────────
                # 1. One-hot: (B, T, C+1)
                y_one_hot = F.one_hot(label, num_classes=C + 1).float()

                # 2. Active classes, permuted for conv1d: (B, C, T)
                y_active = y_one_hot[:, :, 1:].permute(0, 2, 1)

                # 3. Depthwise temporal smoothing
                y_active_smooth = F.conv1d(
                    y_active, kernel, padding=padding, groups=C
                )
                y_active_smooth = torch.clamp(y_active_smooth, min=0.0, max=1.0)

                # 4. Recompute background
                y_bg = (1.0 - y_active_smooth.sum(dim=1, keepdim=True)).clamp(min=0.0)

                # 5. Concatenate → (B, C+1, T)
                y_smooth = torch.cat([y_bg, y_active_smooth], dim=1)

                # 6. Frame-wise normalisation (handles overlapping action curves)
                y_smooth = y_smooth / y_smooth.sum(dim=1, keepdim=True)

                # 7. Permute back → (B*T, C+1)  [used if soft-label loss is needed]
                y_smooth = y_smooth.permute(0, 2, 1).reshape(-1, C + 1)
                # ─────────────────────────────────────────────────────────

                with torch.cuda.amp.autocast():
                    pred  = self._model(frame)
                    pred  = pred.view(-1, self._num_classes + 1)
                    label = label.view(-1)
                    loss  = F.cross_entropy(
                        pred, label, reduction='mean', weight=weights
                    )

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)   # average loss over batches

    def predict(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if seq.dim() == 4:           # (L, C, H, W) → (1, L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)
            return torch.softmax(pred, dim=-1).cpu().numpy()
