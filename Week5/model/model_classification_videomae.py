"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

from transformers import VideoMAEModel 

#Local imports
from model.modules import BaseRGBModel, FCLayers, step

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

            if self._feature_arch == 'videomae_base':
                self._features = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
                self._d = self._features.config.hidden_size 
                
                # Freeze everything
                for param in self._features.parameters():
                    param.requires_grad = False

                # Unfreeze last N layers
                layers_to_unfreeze = 2
                for layer in self._features.encoder.layer[-layers_to_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True

                # Unfreeze final layernorm
                if hasattr(self._features, 'layernorm'):
                    for param in self._features.layernorm.parameters():
                        param.requires_grad = True
                        
            else:
                raise NotImplementedError(f"Architecture {self._feature_arch} not supported.")

            # MLP for classification
            self._fc = FCLayers(self._d, args.num_classes)

            # Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            # Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) 
            ])

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1

            # Sample 16 frames evenly from the input sequence
            if x.shape[1] > 16:
                # Create 16 evenly spaced indices from 0 to the sequence length
                indices = torch.linspace(0, x.shape[1] - 1, 16).long().to(x.device)
                x = x[:, indices, :, :, :] # Slice the temporal dimension
            
            # Center crop to 224x224 if width is larger
            if x.shape[4] > 224:
                w_offset = (x.shape[4] - 224) // 2
                x = x[:, :, :, :, w_offset : w_offset + 224]
            
            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) 
                        
            outputs = self._features(pixel_values=x)

            # Extract the spatio-temporal tokens
            sequence_output = outputs.last_hidden_state #(Batch, Num_Tokens, Hidden_Size)

            # Global Average Pooling over the sequence dimension
            im_feat = sequence_output.mean(dim=1) # B, D

            #MLP Multi-label head
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
                    loss = F.binary_cross_entropy_with_logits(
                            pred, label)

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