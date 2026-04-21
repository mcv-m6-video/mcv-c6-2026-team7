"""
Advanced pooling strategies for temporal U-Net.

Supported pool_type values:
  - 'max'           : Standard MaxPool1d (original baseline)
  - 'avg'           : Standard AvgPool1d
  - 'lp'            : Lp / RMS Pooling  (p=2 by default, learnable optional)
  - 'concat'        : Adaptive Max-Average Pooling (concatenates both; doubles channels)
  - 'stochastic'    : Stochastic Pooling (probabilistic sampling during training)
  - 'blurpool'      : BlurPool (anti-aliased, shift-invariant downsampling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  Lp Pooling  (generalised / RMS pooling)
# ---------------------------------------------------------------------------

class LpPool1d(nn.Module):
    """
    Lp pooling over a 1-D sequence.

    formula: ( mean(x^p) )^(1/p)   over the kernel window

    p=1  →  Average Pooling
    p=2  →  RMS / "energy" Pooling   ← good default for action detection
    p→∞  →  Max Pooling

    Args:
        kernel_size: pooling window size
        p          : the Lp norm exponent (default 2.0 for RMS)
        learnable_p: if True, p becomes a learnable scalar parameter
        stride     : defaults to kernel_size (halving)
    """

    def __init__(self, kernel_size: int = 2, p: float = 2.0,
                 learnable_p: bool = False, stride: int | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

        if learnable_p:
            # initialise at p, but keep it > 1 via softplus during forward
            self._log_p = nn.Parameter(torch.tensor(float(p)).log())
            self.learnable_p = True
        else:
            self.register_buffer("_p", torch.tensor(float(p)))
            self.learnable_p = False

    @property
    def p(self) -> torch.Tensor:
        if self.learnable_p:
            # softplus ensures p stays positive; +1 keeps it ≥ 1
            return 1.0 + F.softplus(self._log_p)
        return self._p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        p = self.p
        # unfold: (B, C, n_windows, kernel_size)
        unfolded = x.unfold(-1, self.kernel_size, self.stride)
        # clamp to avoid NaN from negative values raised to non-integer p
        unfolded = unfolded.abs().clamp(min=1e-8)
        return (unfolded.pow(p).mean(dim=-1)).pow(1.0 / p)

    def extra_repr(self) -> str:
        return (f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"learnable_p={self.learnable_p}")


# ---------------------------------------------------------------------------
# 2.  Concatenated Max-Average Pooling
# ---------------------------------------------------------------------------

class ConcatPool1d(nn.Module):
    """
    Concatenate MaxPool1d and AvgPool1d outputs along the channel dimension.

    ⚠  This DOUBLES the number of channels.  The caller must account for this
       when defining subsequent conv layers.  Use `out_channels_factor = 2`.

    Intuition for action detection:
        Max branch  → "was there a peak intensity frame in this window?"
        Avg branch  → "what was the general movement context?"
    """

    def __init__(self, kernel_size: int = 2, stride: int | None = None):
        super().__init__()
        s = stride or kernel_size
        self.max_pool = nn.MaxPool1d(kernel_size, stride=s)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)

    @staticmethod
    def channel_multiplier() -> int:
        return 2


# ---------------------------------------------------------------------------
# 3.  Stochastic Pooling
# ---------------------------------------------------------------------------

class StochasticPool1d(nn.Module):
    """
    Stochastic pooling (Zeiler & Fergus, 2013) for 1-D sequences.

    Training : sample a position within each window proportional to the
               softmax of the absolute activations.
    Inference: weighted average (= expected value under the same distribution).

    Why for action detection:
        Prevents over-fitting to a single "peak" frame; encourages the U-Net to
        learn that actions span a temporal window, not one magic frame.
    """

    def __init__(self, kernel_size: int = 2, stride: int | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        unfolded = x.unfold(-1, self.kernel_size, self.stride)
        # (B, C, n_windows, kernel_size)

        weights = F.softmax(unfolded.abs(), dim=-1)  # prob distribution

        if self.training:
            # Sample one index per window
            idx = torch.multinomial(
                weights.view(-1, self.kernel_size), num_samples=1
            ).view(*unfolded.shape[:-1], 1)
            pooled = unfolded.gather(-1, idx).squeeze(-1)
        else:
            # Weighted average at inference (= E[x] under the distribution)
            pooled = (unfolded * weights).sum(dim=-1)

        return pooled

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}"


# ---------------------------------------------------------------------------
# 4.  BlurPool (anti-aliased downsampling)
# ---------------------------------------------------------------------------

class BlurPool1d(nn.Module):
    """
    BlurPool for 1-D sequences (Zhang, 2019: "Making CNNs Shift-Invariant Again").

    Pipeline: MaxPool (stride-1) → depthwise Gaussian blur → stride-2 subsample

    Why for action detection:
        Standard MaxPool is *not* shift-invariant: if an action starts one frame
        later, the pooled output can change drastically.  BlurPool smooths this
        out, making start/end-time predictions much more stable.

    Args:
        channels   : number of input channels (needed for depthwise conv)
        kernel_size: blur kernel size (3 or 5 recommended)
        stride     : downsampling stride (default 2)
        sigma      : Gaussian sigma (None = auto from kernel_size)
    """

    def __init__(self, channels: int, kernel_size: int = 3,
                 stride: int = 2, sigma: float | None = None):
        super().__init__()
        self.stride = stride

        # Build 1-D Gaussian kernel
        if sigma is None:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8  # OpenCV formula
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = kernel / kernel.sum()

        # (out_ch, in_ch/groups, kW)  — depthwise: groups = channels
        kernel = kernel.view(1, 1, kernel_size).repeat(channels, 1, 1)
        self.register_buffer("kernel", kernel)

        self.channels = channels
        self.pad = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise blur, then subsample
        blurred = F.conv1d(
            x, self.kernel,
            stride=1, padding=self.pad, groups=self.channels
        )
        return blurred[:, :, ::self.stride]

    def extra_repr(self) -> str:
        return f"channels={self.channels}, stride={self.stride}"


class MaxBlurPool1d(nn.Module):
    """
    Convenience wrapper: MaxPool1d(stride=1) followed by BlurPool1d.
    This is the canonical BlurPool as described in the paper.
    """

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)
        self.blur = BlurPool1d(channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blur(self.max_pool(x))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

POOL_TYPES = ('max', 'avg', 'lp', 'concat', 'stochastic', 'blurpool')


def make_pool(pool_type: str, channels: int, kernel_size: int = 2) -> nn.Module:
    """
    Factory that returns the requested pooling module.

    Args:
        pool_type  : one of POOL_TYPES
        channels   : number of input channels  (needed by BlurPool & ConcatPool)
        kernel_size: pooling window / stride (default 2 → halve the sequence)

    Returns:
        An nn.Module with a standard forward(x) signature.

    Note on ConcatPool:
        It doubles the channel dimension.  Query `ConcatPool1d.channel_multiplier()`
        and multiply downstream channel counts accordingly.
    """
    pool_type = pool_type.lower()
    if pool_type == 'max':
        return nn.MaxPool1d(kernel_size)
    elif pool_type == 'avg':
        return nn.AvgPool1d(kernel_size)
    elif pool_type == 'lp':
        return LpPool1d(kernel_size, p=2.0, learnable_p=False)
    elif pool_type == 'concat':
        return ConcatPool1d(kernel_size)
    elif pool_type == 'stochastic':
        return StochasticPool1d(kernel_size)
    elif pool_type == 'blurpool':
        return MaxBlurPool1d(channels, kernel_size=3, stride=kernel_size)
    else:
        raise ValueError(
            f"Unknown pool_type '{pool_type}'. Choose from: {POOL_TYPES}"
        )