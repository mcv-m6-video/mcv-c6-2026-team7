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
            
            elif self._feature_arch == 'convnextv2_pico':
                # Setting num_classes=0 automatically strips the final classification head 
                features = timm.create_model('convnextv2_pico.fcmae_ft_in1k', pretrained=True, num_classes=0)
                self._d = features.num_features

            else:
                raise NotImplementedError(args._feature_arch)

            self._features = features

            # Temporal transformer
            self._pos_embed = nn.Parameter(torch.randn(1, args.clip_len, self._d) * 0.02)
            dropout = getattr(args, 'transformer_dropout', 0.25)
            num_layers = getattr(args, 'transformer_layers', 3)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self._d, 
                nhead=8, 
                dim_feedforward=self._d * 2,
                dropout=dropout, 
                batch_first=True, 
                norm_first=True
            )
            self._temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # MLP for classification
            self._fc = FCLayers(self._d, args.num_classes+1) # +1 for background class (we now perform per-frame classification with softmax, therefore we have the extra background class)

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

            # Temporal transformer
            im_feat = im_feat + self._pos_embed
            im_feat = self._temporal(im_feat) #B, T, D

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

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()
        
        # --- GAUSSIAN KERNEL FOR TEMPORAL SMOOTHING ---
        window_size = 5
        sigma = 0.55
        C = self._num_classes
        
        # 1D Gaussian kernel
        x_coords = torch.arange(-window_size // 2 + 1., window_size // 2 + 1., device=self.device)
        gauss = torch.exp(-x_coords**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        
        # Reshape for depthwise 1D convolution
        kernel = gauss.view(1, 1, window_size).repeat(C, 1, 1)
        padding = window_size // 2
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
                
                # Prevent numerical overflow in edge cases (e.g., overlapping events)
                y_active_smooth = torch.clamp(y_active_smooth, min=0.0, max=1.0)

                # 4. Recompute background probability (1.0 minus the sum of active probabilities)
                y_bg = 1.0 - y_active_smooth.sum(dim=1, keepdim=True)
                y_bg = torch.clamp(y_bg, min=0.0, max=1.0)

                # 5. Concatenate background and smoothed active classes back together: (B, C+1, T)
                y_smooth = torch.cat([y_bg, y_active_smooth], dim=1)

                # 6. Permute back to (B, T, C+1) and flatten into (B*T, C+1) for the loss function
                y_smooth = y_smooth.permute(0, 2, 1).reshape(-1, C + 1)
                # --------------------------------------

                # --- DEBUG PRINT CHECK ---
                # if batch_idx == 0:
                #     print("\n" + "="*50)
                #     print(" TEMPORAL SMOOTHING DEBUG CHECK (First sequence)")
                #     print("="*50)
                    
                #     # Reshape back to (B, T, C+1) to look at the first sample
                #     # frame.shape[1] is clip_len (T)
                #     T = frame.shape[1]
                #     sample_smooth = y_smooth.view(-1, T, C + 1)[0].cpu().numpy()
                #     sample_raw = label[0].cpu().numpy()
                    
                #     # Print frame by frame (limit to first 15 frames to keep the console clean)
                #     print(f"Showing first {min(15, T)} frames:")
                #     print(f"{'Frame':<7} | {'Raw Label':<10} | {'Smoothed Probabilities (Bg, C1, C2...)':<40}")
                #     print("-" * 75)
                    
                #     for t in range(min(15, T)):
                #         probs_str = ", ".join([f"{p:.3f}" for p in sample_smooth[t]])
                #         # Add a little marker '*' if an active event is happening
                #         marker = " *" if sample_raw[t] > 0 else ""
                #         print(f"  {t:02d}{marker:<4} | Class {sample_raw[t]:<4} | [{probs_str}]")
                        
                #     print("="*50 + "\n")
                # -------------------------

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1) # B*T, num_classes + 1
                    
                    # Compute cross entropy using the soft labels
                    # PyTorch handles target probabilities automatically if shapes match
                    loss = F.cross_entropy(
                            pred, y_smooth, reduction='mean', weight=weights)

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
        