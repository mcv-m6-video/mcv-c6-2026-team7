"""
model_bigru_traj.py
===================
Versión del modelo BiGRU con integración de trayectorias de balón y jugadores.

Cambios respecto al original:
  1. TrajectoryFusionEncoder proyecta features de trayectoria (139-dim)
     al espacio del backbone (self._d) y las SUMA a los tokens visuales
     ANTES del BiGRU → el BiGRU ya modela la dinámica temporal
     sobre features enriquecidas.

  2. BallAttentionGate pondera los tokens por la velocidad del balón
     DESPUÉS del BiGRU → actúa como filtro sobre los tokens ya
     contextualizados temporalmente.

  3. El forward() acepta un argumento opcional `traj` (B, T, 139).
     Si no se pasa (o es None), el modelo se comporta exactamente
     igual que el original → compatible con código existente.

  4. El método epoch() y predict() aceptan opcionalmente traj_features
     en el batch dict bajo la clave 'traj'.

USO:
    # En config JSON añadir:
    #   "model_module": "model_bigru_traj",
    #   "use_traj": true,
    #   "traj_dir": "/path/to/traj_features",
    #   "traj_fusion": "add"   (o "concat")

    # Extraer trayectorias primero:
    #   python trajectory_pipeline.py --mode extract ...
"""

# Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

# Local imports
from model.modules import BaseRGBModel, FCLayers, step

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DEL PIPELINE DE TRAYECTORIAS
# (deben coincidir con trajectory_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────
BALL_FEAT_DIM   = 7    # [x, y, vx, vy, ax, ay, visible]
PLAYER_FEAT_DIM = 6    # [x, y, vx, vy, dist_ball, team]
MAX_PLAYERS     = 22
TRAJ_FEAT_DIM   = BALL_FEAT_DIM + PLAYER_FEAT_DIM * MAX_PLAYERS  # 139


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULOS DE TRAYECTORIA
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryFusionEncoder(nn.Module):
    """
    Proyecta las features de trayectoria al espacio del backbone (visual_dim)
    y las suma o concatena a los tokens visuales.

    Fusión aditiva (fusion_mode='add'):
        tokens_out = tokens_visual + MLP(traj_features)
        → Sin cambio de dimensión, zero overhead en el resto del modelo.

    Fusión concatenada (fusion_mode='concat'):
        tokens_out = Linear(concat([tokens_visual, MLP(traj_features)]))
        → Más expresivo, útil si add no mejora.
    """
    def __init__(self, traj_dim: int, visual_dim: int,
                 hidden_dim: int = 128, fusion_mode: str = 'add',
                 dropout: float = 0.1):
        super().__init__()
        self.fusion_mode = fusion_mode

        self.projector = nn.Sequential(
            nn.Linear(traj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, visual_dim),
            nn.LayerNorm(visual_dim),
        )
        if fusion_mode == 'concat':
            self.fusion_proj = nn.Linear(visual_dim * 2, visual_dim)

    def forward(self, visual_tokens: torch.Tensor,
                traj_features: torch.Tensor) -> torch.Tensor:
        """
        visual_tokens : (B, T, D)
        traj_features : (B, T, TRAJ_FEAT_DIM)
        returns       : (B, T, D)
        """
        traj_proj = self.projector(traj_features)          # (B, T, D)
        if self.fusion_mode == 'add':
            return visual_tokens + traj_proj
        else:  # concat
            return self.fusion_proj(
                torch.cat([visual_tokens, traj_proj], dim=-1))


class BallAttentionGate(nn.Module):
    """
    Gate temporal basado en la dinámica del balón.
    Frames donde el balón se mueve rápido (pre-acción) reciben mayor peso.

    El gate es un escalar por frame, aprendido desde las 7 features del balón.
    Se aplica como multiplicación elemento a elemento sobre los tokens.
    """
    def __init__(self, visual_dim: int, ball_feat_dim: int = BALL_FEAT_DIM):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(ball_feat_dim, visual_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(visual_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor,
                traj_features: torch.Tensor) -> torch.Tensor:
        """
        tokens        : (B, T, D)
        traj_features : (B, T, TRAJ_FEAT_DIM)
        returns       : (B, T, D)
        """
        ball_feats = traj_features[..., :BALL_FEAT_DIM]    # (B, T, 7)
        attn = self.gate(ball_feats)                        # (B, T, 1)
        return tokens * attn                                # broadcast → (B, T, D)


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE BiGRU RESIDUAL (igual que el original)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBiGRU(nn.Module):
    """
    RNN block using a Bidirectional GRU with Transformer-style
    Layer Normalization and Residual connections.
    """
    def __init__(self, dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm    = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out, _ = self.gru(x)
        out = self.dropout(out)
        return self.norm(residual + out)


# ─────────────────────────────────────────────────────────────────────────────
# MODELO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            self._feature_arch = args.feature_arch
            self._use_traj = getattr(args, 'use_traj', False)
            traj_fusion   = getattr(args, 'traj_fusion', 'add')

            # ── Backbone visual (sin cambios) ──────────────────────────────
            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()
                self._d = feat_dim

            elif self._feature_arch == 'convnextv2_pico':
                features = timm.create_model(
                    'convnextv2_pico.fcmae_ft_in1k', pretrained=True, num_classes=0)
                self._d = features.num_features
            else:
                raise NotImplementedError(self._feature_arch)

            self._features = features

            # ── [NUEVO] Módulos de trayectoria ─────────────────────────────
            # Se insertan ANTES del BiGRU para que el GRU ya vea tokens
            # enriquecidos con información cinemática.
            if self._use_traj:
                self._traj_encoder = TrajectoryFusionEncoder(
                    traj_dim=TRAJ_FEAT_DIM,
                    visual_dim=self._d,
                    hidden_dim=min(256, self._d // 2),
                    fusion_mode=traj_fusion,
                    dropout=0.1,
                )
                self._ball_gate = BallAttentionGate(
                    visual_dim=self._d,
                    ball_feat_dim=BALL_FEAT_DIM,
                )
                traj_params = (
                    sum(p.numel() for p in self._traj_encoder.parameters()) +
                    sum(p.numel() for p in self._ball_gate.parameters())
                )
                print(f'[TrajModel] Módulos de trayectoria activos. '
                      f'Parámetros añadidos: {traj_params:,}  '
                      f'(fusion_mode={traj_fusion})')

            # ── Temporal modelling via Residual BiGRU (sin cambios) ────────
            self._temporal = ResidualBiGRU(dim=self._d, num_layers=2, dropout=0.3)

            # ── MLP para clasificación (sin cambios) ───────────────────────
            self._fc = FCLayers(self._d, args.num_classes + 1)

            # ── Augmentations (sin cambios) ────────────────────────────────
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])
            self.standarization = T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
            ])

        def forward(self, x, traj=None):
            """
            x    : (B, T, C, H, W)  frames del clip
            traj : (B, T, 139)      features de trayectoria  [OPCIONAL]

            Si traj es None el modelo se comporta exactamente igual
            que el original (retro-compatible).
            """
            x = self.normalize(x)
            batch_size, clip_len, channels, height, width = x.shape

            if self.training:
                x = self.augment(x)
            x = self.standarize(x)

            # ── 1. Backbone visual ──────────────────────────────────────────
            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._d)          # (B, T, D)

            # ── 2. [NUEVO] Fusión de trayectorias (pre-GRU) ─────────────────
            # Insertar aquí para que el BiGRU modele la dinámica temporal
            # sobre tokens ya enriquecidos con posición/velocidad.
            if self._use_traj and traj is not None:
                im_feat = self._traj_encoder(im_feat, traj)   # (B, T, D)

            # ── 3. Temporal modelling BiGRU ─────────────────────────────────
            im_feat = self._temporal(im_feat)                  # (B, T, D)

            # ── 4. [NUEVO] Ball attention gate (post-GRU) ───────────────────
            # Ponderar tokens ya contextualizados por la actividad del balón.
            # Esto es especialmente útil para predicción: los frames justo
            # antes de una acción (balón acelerando) reciben más peso.
            if self._use_traj and traj is not None:
                im_feat = self._ball_gate(im_feat, traj)       # (B, T, D)

            # ── 5. MLP ──────────────────────────────────────────────────────
            im_feat = self._fc(im_feat)                        # (B, T, C+1)

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
            total = sum(p.numel() for p in self.parameters())
            print(f'Model params: {total:,}')

    # ── Wrapper externo (igual que el original + soporte traj) ────────────────

    def __init__(self, args=None):
        self.device = 'cpu'
        if torch.cuda.is_available() and \
                getattr(args, 'device', 'cpu') == 'cuda':
            self.device = 'cuda'

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args
        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        # ── Gaussian kernel para temporal label smoothing (sin cambios) ───
        window_size = 5
        sigma = 0.55
        C = self._num_classes

        x_coords = torch.arange(
            -window_size // 2 + 1., window_size // 2 + 1., device=self.device)
        gauss = torch.exp(-x_coords ** 2 / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel  = gauss.view(1, 1, window_size).repeat(C, 1, 1)
        padding = window_size // 2

        weights = torch.tensor(
            [1.0] + [5.0] * self._num_classes,
            dtype=torch.float32, device=self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch in tqdm(loader):
                frame = batch['frame'].to(self.device).float()
                label = batch['label'].to(self.device).long()

                # ── Trayectorias (opcional) ────────────────────────────────
                # El DataLoader debe incluir 'traj' si use_traj=True.
                # Si no existe la clave, se pasa None (modo degradado).
                traj = None
                if 'traj' in batch:
                    traj = batch['traj'].to(self.device).float()  # (B, T, 139)

                # ── Temporal label smoothing (sin cambios) ─────────────────
                y_one_hot     = F.one_hot(label, num_classes=C + 1).float()
                y_active      = y_one_hot[:, :, 1:].permute(0, 2, 1)
                y_active_smooth = F.conv1d(
                    y_active, kernel, padding=padding, groups=C)
                y_active_smooth = torch.clamp(y_active_smooth, 0.0, 1.0)
                y_bg    = torch.clamp(
                    1.0 - y_active_smooth.sum(dim=1, keepdim=True), 0.0, 1.0)
                y_smooth = torch.cat([y_bg, y_active_smooth], dim=1)
                y_smooth = y_smooth.permute(0, 2, 1).reshape(-1, C + 1)

                with torch.cuda.amp.autocast():
                    pred = self._model(frame, traj=traj)           # ← traj aquí
                    pred = pred.view(-1, self._num_classes + 1)
                    loss = F.cross_entropy(
                        pred, y_smooth, reduction='mean', weight=weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq, traj=None):
        """
        seq  : numpy array o Tensor (L, C, H, W) o (1, L, C, H, W)
        traj : numpy array o Tensor (L, 139) o (1, L, 139)  [opcional]
        """
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:
            seq = seq.unsqueeze(0)
        seq = seq.to(self.device).float()

        traj_t = None
        if traj is not None:
            if not isinstance(traj, torch.Tensor):
                traj_t = torch.FloatTensor(traj)
            else:
                traj_t = traj
            if len(traj_t.shape) == 2:
                traj_t = traj_t.unsqueeze(0)
            traj_t = traj_t.to(self.device).float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq, traj=traj_t)
            pred = torch.softmax(pred, dim=-1)
            return pred.cpu().numpy()
