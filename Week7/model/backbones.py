"""
Backbone and adapter utilities for Week7 spotting models.
"""

import importlib
import inspect
from typing import Any, Tuple, cast

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


_REGNET_BACKBONES = {
    'rny002': 'regnety_002',
    'rny004': 'regnety_004',
    'rny008': 'regnety_008',
}

_CONVNEXT_BACKBONES = {
    'convnextv2_pico': 'convnextv2_pico.fcmae_ft_in1k',
    'convnextv2_atto': 'convnextv2_atto.fcmae_ft_in1k',
}

_X3D_BACKBONES = {
    'x3d_s',
    'x3d_m',
}


def available_backbones():
    return sorted(
        list(_REGNET_BACKBONES.keys())
        + list(_CONVNEXT_BACKBONES.keys())
        + sorted(_X3D_BACKBONES)
    )


def _use_pretrained_backbone(args) -> bool:
    return bool(getattr(args, 'backbone_pretrained', True))


def _freeze_backbone(args) -> bool:
    return bool(getattr(args, 'freeze_backbone', False))


class FrameBackboneAdapter(nn.Module):
    """Adapter for frame-wise 2D backbones to produce clip features BxTxD."""

    def __init__(self, model: nn.Module, feat_dim: int):
        super().__init__()
        self.model = model
        self.feat_dim = feat_dim

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        batch_size, clip_len, channels, height, width = clips.shape
        feats = self.model(clips.reshape(batch_size * clip_len, channels, height, width))
        return feats.reshape(batch_size, clip_len, self.feat_dim)


class X3DBackboneAdapter(nn.Module):
    """Adapter for X3D backbones to produce clip features BxTxD."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.feat_dim = self._infer_feat_dim()

    @staticmethod
    def _is_head_block(module: nn.Module) -> bool:
        name = module.__class__.__name__.lower()
        return any(token in name for token in ('head', 'pool', 'projectedpool', 'classification'))

    def _extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, 'blocks'):
            raise RuntimeError('X3D model does not expose a blocks attribute for feature extraction.')

        blocks_attr = getattr(self.model, 'blocks')
        if not isinstance(blocks_attr, (nn.ModuleList, list, tuple)):
            raise RuntimeError('X3D model blocks attribute is not an iterable module container.')

        blocks = list(blocks_attr)
        stop_idx = len(blocks)
        if stop_idx > 0 and self._is_head_block(blocks[-1]):
            stop_idx -= 1

        for block_idx in range(stop_idx):
            x = blocks[block_idx](x)

        return x

    def _infer_feat_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 8, 3, 112, 112)
            feats = self.forward(dummy)
        return int(feats.shape[-1])

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        input_len = clips.shape[1]

        # X3D expects BxCxTxHxW input.
        x = clips.permute(0, 2, 1, 3, 4)
        x = self._extract_feature_map(x)

        # Keep temporal resolution and pool only spatial dimensions.
        x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)

        if x.shape[1] != input_len:
            x = F.interpolate(
                x.transpose(1, 2),
                size=input_len,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)

        if x.shape[1] != input_len:
            raise RuntimeError(
                'Failed to align X3D temporal length to input clip length (expected {}, got {}).'.format(
                    input_len,
                    x.shape[1],
                )
            )

        return x


def _create_torchvision_x3d(backbone_name: str, pretrained: bool) -> nn.Module | None:
    try:
        import torchvision.models.video as tv_video
    except Exception:
        return None

    builder = getattr(tv_video, backbone_name, None)
    if builder is None:
        return None

    signature = inspect.signature(builder)
    if 'weights' in signature.parameters:
        weights = None
        if pretrained:
            weights_enum_name = '{}_Weights'.format(backbone_name.upper())
            weights_enum = getattr(tv_video, weights_enum_name, None)
            if weights_enum is not None and hasattr(weights_enum, 'DEFAULT'):
                weights = weights_enum.DEFAULT
        return cast(nn.Module, builder(weights=weights))

    if 'pretrained' in signature.parameters:
        return cast(nn.Module, builder(pretrained=pretrained))

    return cast(nn.Module, builder())


def _create_pytorchvideo_x3d(backbone_name: str, pretrained: bool) -> nn.Module | None:
    try:
        hub_module = importlib.import_module('pytorchvideo.models.hub')
    except Exception:
        return None

    builder = getattr(hub_module, backbone_name, None)
    if builder is None:
        return None

    return cast(nn.Module, builder(pretrained=pretrained))


def _create_x3d_model(backbone_name: str, pretrained: bool):
    model = _create_torchvision_x3d(backbone_name, pretrained)
    if model is not None:
        return model

    model = _create_pytorchvideo_x3d(backbone_name, pretrained)
    if model is not None:
        return model

    try:
        return cast(nn.Module, torch.hub.load('facebookresearch/pytorchvideo', backbone_name, pretrained=pretrained))
    except Exception as exc:
        raise RuntimeError(
            'Unable to build {}. Tried torchvision, local pytorchvideo package, and torch.hub fallback. '
            'Install X3D dependencies with: pip install pytorchvideo fvcore iopath. '
            'If the server is offline, keep backbone_pretrained=false to avoid downloading pretrained weights. '
            'torch.hub error: {}'.format(backbone_name, exc)
        ) from exc


def create_backbone(args) -> Tuple[nn.Module, int]:
    """Build a configured backbone and return the feature extractor with output dimension."""
    backbone_name = getattr(args, 'feature_arch', None)
    if backbone_name is None:
        raise ValueError('Missing required config key: feature_arch')

    if backbone_name in _REGNET_BACKBONES:
        model = timm.create_model(
            _REGNET_BACKBONES[backbone_name],
            pretrained=_use_pretrained_backbone(args),
        )
        regnet_model = cast(Any, model)
        feat_dim = int(regnet_model.head.fc.in_features)
        regnet_model.head.fc = nn.Identity()
        features = FrameBackboneAdapter(cast(nn.Module, regnet_model), feat_dim)
    elif backbone_name in _CONVNEXT_BACKBONES:
        model = timm.create_model(
            _CONVNEXT_BACKBONES[backbone_name],
            pretrained=_use_pretrained_backbone(args),
            num_classes=0,
        )
        convnext_model = cast(Any, model)
        feat_dim = int(convnext_model.num_features)
        features = FrameBackboneAdapter(cast(nn.Module, convnext_model), feat_dim)
    elif backbone_name in _X3D_BACKBONES:
        model = cast(nn.Module, _create_x3d_model(backbone_name, _use_pretrained_backbone(args)))
        features = X3DBackboneAdapter(model)
        feat_dim = features.feat_dim
    else:
        raise NotImplementedError(
            'Unsupported backbone "{}". Available: {}'.format(
                backbone_name,
                ', '.join(available_backbones()),
            )
        )

    if _freeze_backbone(args):
        for param in features.parameters():
            param.requires_grad = False

    return features, feat_dim
