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


def focal_loss(pred, label, gamma=2.0):
    """Binary focal loss for multi-label classification.
    Down-weights easy examples (high pt) so the model focuses on hard ones.
    """
    bce = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
    pt = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

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

            # Temporal encoder
            self._pos_embed = nn.Parameter(torch.randn(1, args.clip_len + 1, self._d) * 0.02)
            self._cls_token = nn.Parameter(torch.randn(1, 1, self._d) * 0.02)
            dropout = getattr(args, 'transformer_dropout', 0.25)
            num_layers = getattr(args, 'transformer_layers', 3)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self._d, nhead=8, dim_feedforward=self._d * 2,
                dropout=dropout, batch_first=True, norm_first=True
            )
            self._temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # MLP for classification
            self._fc = FCLayers(self._d, args.num_classes)

            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
                T.RandomGrayscale(p=0.1),
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

            # Temporal Transformer encoder with CLS token
            cls = self._cls_token.expand(batch_size, -1, -1)   # (B, 1, D)
            im_feat = torch.cat([cls, im_feat], dim=1)          # (B, T+1, D)
            im_feat = im_feat + self._pos_embed
            im_feat = self._temporal(im_feat)                   # (B, T+1, D)
            im_feat = im_feat[:, 0, :]                          # CLS token -> (B, D)

            #MLP
            im_feat = self._fc(im_feat) #B, num_classes

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

    def get_optimizer(self, opt_args):
        return torch.optim.AdamW(
            self._model.parameters(),
            lr=opt_args['lr'],
            weight_decay=0.05
        ), torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).float()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    if self._args.loss_type == 'focal':
                        loss = focal_loss(pred, label)
                    else:
                        loss = F.binary_cross_entropy_with_logits(pred, label)

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
            pred = torch.sigmoid(pred)
            
            return pred.cpu().numpy()
