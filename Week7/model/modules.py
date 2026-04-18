"""
File containing the different modules related to the model: T-DEED.
"""

#Standard imports
import abc
from typing import Tuple

import timm
import torch
import torch.nn as nn

#Local imports

_REGNET_BACKBONES = {
    'rny002': 'regnety_002',
    'rny004': 'regnety_004',
    'rny008': 'regnety_008',
}

_CONVNEXT_BACKBONES = {
    'convnextv2_pico': 'convnextv2_pico.fcmae_ft_in1k',
    'convnextv2_atto': 'convnextv2_atto.fcmae_ft_in1k',
}


def available_backbones():
    return sorted(list(_REGNET_BACKBONES.keys()) + list(_CONVNEXT_BACKBONES.keys()))


def _use_pretrained_backbone(args) -> bool:
    return bool(getattr(args, 'backbone_pretrained', True))


def _freeze_backbone(args) -> bool:
    return bool(getattr(args, 'freeze_backbone', False))


def create_backbone(args) -> Tuple[nn.Module, int]:
    """Build a configured backbone and return the feature extractor with output dimension."""
    backbone_name = getattr(args, 'feature_arch', None)
    if backbone_name is None:
        raise ValueError('Missing required config key: feature_arch')

    if backbone_name in _REGNET_BACKBONES:
        features = timm.create_model(
            _REGNET_BACKBONES[backbone_name],
            pretrained=_use_pretrained_backbone(args),
        )
        feat_dim = features.head.fc.in_features
        features.head.fc = nn.Identity()
    elif backbone_name in _CONVNEXT_BACKBONES:
        features = timm.create_model(
            _CONVNEXT_BACKBONES[backbone_name],
            pretrained=_use_pretrained_backbone(args),
            num_classes=0,
        )
        feat_dim = features.num_features
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

class ABCModel:

    @abc.abstractmethod
    def get_optimizer(self, opt_args):
        raise NotImplementedError()

    @abc.abstractmethod
    def epoch(self, loader, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, seq):
        raise NotImplementedError()

    @abc.abstractmethod
    def state_dict(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, state_dict):
        raise NotImplementedError()

class BaseRGBModel(ABCModel):

    def get_optimizer(self, opt_args):
        return torch.optim.AdamW(self._get_params(), **opt_args), \
            torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

    """ Assume there is a self._model """

    def _get_params(self):
        return list(self._model.parameters())

    def state_dict(self):
        if isinstance(self._model, nn.DataParallel):
            return self._model.module.state_dict()
        return self._model.state_dict()

    def load(self, state_dict):
        if isinstance(self._model, nn.DataParallel):
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)    

class FCLayers(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        if len(x.shape) == 3:
            b, t, d = x.shape
            x = x.reshape(b * t, d)
            return self._fc_out(self.dropout(x)).reshape(b, t, -1)
        elif len(x.shape) == 2:
            return self._fc_out(self.dropout(x))

def step(optimizer, scaler, loss, lr_scheduler=None):
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    optimizer_was_stepped = True
    if scaler is None:
        optimizer.step()
    else:
        prev_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_was_stepped = scaler.get_scale() >= prev_scale

    if lr_scheduler is not None and optimizer_was_stepped:
        lr_scheduler.step()
    optimizer.zero_grad()
    