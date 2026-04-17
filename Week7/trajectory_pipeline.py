"""
trajectory_pipeline.py
=======================
Pipeline de extracción de trayectorias de balón y jugadores.
Compatible con arquitectura BiGRU / T-DEED para SoccerNet BAS.

ESTRUCTURA DE DATOS ESPERADA:
  frame_dir/
    england_efl/
      2019-2020/
        2019-10-01 - Blackburn Rovers - Nottingham Forest/
          frame00001.jpg
          frame00002.jpg
          ...

DUAL-MODEL APPROACH:
  - Balón    → ball_detection_model.pt  (fine-tuned, clase 0)
  - Jugadores → yolov8m.pt             (COCO, clase 0 = person)

SALIDA:
  output_dir/
    england_efl_2019-2020_2019-10-01 - Blackburn Rovers - Nottingham Forest.npy
    ...
  Cada .npy tiene shape (total_frames, 139), indexado por número de frame.

USO:
  python trajectory_pipeline.py --mode extract \
      --frame_dir /data-fast/.../SN-BAS-2025_frames/398x224 \
      --output_dir /data-fast/.../SN-BAS-2025_traj_features \
      --ball_model ../ball_detection_model.pt \
      --player_model yolov8m.pt

  python trajectory_pipeline.py --mode visualize \
      --game_dir "/data-fast/.../398x224/england_efl/2019-2020/2019-10-01 - Blackburn Rovers - Nottingham Forest" \
      --traj_npy /data-fast/.../traj_features/england_efl_2019-2020_2019-10-01 - Blackburn Rovers - Nottingham Forest.npy

  python trajectory_pipeline.py --mode info

REQUISITOS:
  pip install ultralytics opencv-python-headless numpy torch
"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

BALL_MODEL_CLASS  = 0   # clase 0 en modelo especializado = ball
COCO_PERSON_CLASS = 0   # clase 0 en COCO = person

MAX_PLAYERS     = 22
BALL_FEAT_DIM   = 7    # [x, y, vx, vy, ax, ay, visible]
PLAYER_FEAT_DIM = 6    # [x, y, vx, vy, dist_ball, team_proxy]
TRAJ_FEAT_DIM   = BALL_FEAT_DIM + PLAYER_FEAT_DIM * MAX_PLAYERS  # 139


# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_neighbor_velocity(
    current:  np.ndarray,
    previous: np.ndarray,
    max_dist: float = 0.15,
) -> np.ndarray:
    velocities = np.zeros((len(current), 2))
    if len(previous) == 0 or len(current) == 0:
        return velocities
    for i, c in enumerate(current):
        dists = np.linalg.norm(previous - c, axis=1)
        j = np.argmin(dists)
        if dists[j] < max_dist:
            velocities[i] = c - previous[j]
    return velocities


def game_dir_to_key(game_dir: Path, frame_dir: Path) -> str:
    """
    Convierte la ruta de un partido a una clave plana para el .npy.
    Ejemplo:
      england_efl/2019-2020/2019-10-01 - Blackburn Rovers - Nottingham Forest
      → england_efl_2019-2020_2019-10-01 - Blackburn Rovers - Nottingham Forest
    """
    rel = game_dir.relative_to(frame_dir)
    parts = rel.parts  # ('england_efl', '2019-2020', '2019-10-01 - ...')
    return "_".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTOR DUAL-MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryExtractor:
    """
    Extrae features cinemáticas de balón y jugadores usando dos modelos:
      - ball_model  : YOLOv8 fine-tuned en balón de fútbol  (clase 0 = ball)
      - player_model: YOLOv8m COCO                          (clase 0 = person)
    """

    def __init__(
        self,
        ball_model_path:   str   = "ball_detection_model.pt",
        player_model_path: str   = "yolov8m.pt",
        device:            str   = "cuda" if torch.cuda.is_available() else "cpu",
        conf_ball:         float = 0.25,
        conf_player:       float = 0.40,
    ):
        self.device      = device
        self.conf_ball   = conf_ball
        self.conf_player = conf_player

        print(f"[TrajectoryExtractor] Balón    : {ball_model_path}")
        self.ball_model = YOLO(ball_model_path)
        self.ball_model.to(device)

        print(f"[TrajectoryExtractor] Jugadores: {player_model_path}")
        self.player_model = YOLO(player_model_path)
        self.player_model.to(device)

        print(f"[TrajectoryExtractor] device={device}  "
              f"conf_ball={conf_ball}  conf_player={conf_player}")

    def _get_detections(
        self,
        frame: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Inferencia dual sobre un frame. Devuelve (ball_xy, players_xy)."""
        h, w = frame.shape[:2]

        # ── BALÓN ─────────────────────────────────────────────────────────
        ball_results = self.ball_model(frame, verbose=False)[0]
        ball_box  = None
        best_conf = self.conf_ball
        for box in ball_results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == BALL_MODEL_CLASS and conf > best_conf:
                xyxy      = box.xyxy[0].cpu().numpy()
                ball_box  = np.array([((xyxy[0]+xyxy[2])/2)/w,
                                      ((xyxy[1]+xyxy[3])/2)/h])
                best_conf = conf

        # ── JUGADORES ─────────────────────────────────────────────────────
        player_results = self.player_model(
            frame, classes=[COCO_PERSON_CLASS], verbose=False)[0]
        players = []
        for box in player_results.boxes:
            if float(box.conf[0]) > self.conf_player:
                xyxy = box.xyxy[0].cpu().numpy()
                players.append([((xyxy[0]+xyxy[2])/2)/w,
                                 ((xyxy[1]+xyxy[3])/2)/h])

        player_boxes = np.array(players) if players else np.zeros((0, 2))
        return ball_box, player_boxes

    def extract_from_frames(
        self,
        frames:  List[np.ndarray],
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Extrae features de una lista de frames ordenados.
        Returns: (T, 139)
        """
        T        = len(frames)
        features = np.zeros((T, TRAJ_FEAT_DIM), dtype=np.float32)
        ball_history   = []
        player_history = []

        for t, frame in enumerate(frames):
            if verbose and t % 50 == 0:
                print(f"  [{t:6d}/{T}]", end="\r")

            ball_box, player_boxes = self._get_detections(frame)

            # ── BALÓN ─────────────────────────────────────────────────────
            ball_history.append(ball_box)
            visible = 1.0 if ball_box is not None else 0.0
            bx = ball_box[0] if ball_box is not None else 0.5
            by = ball_box[1] if ball_box is not None else 0.5
            vx_b = vy_b = ax_b = ay_b = 0.0

            if t >= 1 and ball_history[t-1] is not None and ball_box is not None:
                vx_b = bx - ball_history[t-1][0]
                vy_b = by - ball_history[t-1][1]
            if t >= 2:
                pvx = (ball_history[t-1][0] - ball_history[t-2][0]
                       if ball_history[t-1] is not None and ball_history[t-2] is not None else 0.0)
                pvy = (ball_history[t-1][1] - ball_history[t-2][1]
                       if ball_history[t-1] is not None and ball_history[t-2] is not None else 0.0)
                ax_b = vx_b - pvx
                ay_b = vy_b - pvy

            features[t, :BALL_FEAT_DIM] = [bx, by, vx_b, vy_b, ax_b, ay_b, visible]

            # ── JUGADORES ─────────────────────────────────────────────────
            pxy = player_boxes
            player_history.append(pxy)

            if len(pxy) > 0:
                ball_pt      = np.array([bx, by])
                dists        = np.linalg.norm(pxy - ball_pt, axis=1)
                order        = np.argsort(dists)
                pxy_sorted   = pxy[order]
                dists_sorted = dists[order]
            else:
                pxy_sorted   = np.zeros((0, 2))
                dists_sorted = np.zeros(0)

            pv = (_nearest_neighbor_velocity(pxy_sorted, player_history[t-1])
                  if t >= 1 and len(player_history[t-1]) > 0 and len(pxy_sorted) > 0
                  else np.zeros((len(pxy_sorted), 2)))

            team_proxy = (pxy_sorted[:, 1] > 0.5).astype(np.float32) if len(pxy_sorted) > 0 else np.zeros(0)

            for i in range(MAX_PLAYERS):
                base = BALL_FEAT_DIM + i * PLAYER_FEAT_DIM
                if i < len(pxy_sorted):
                    px, py   = pxy_sorted[i]
                    pvx, pvy = (pv[i, 0], pv[i, 1]) if i < len(pv) else (0.0, 0.0)
                    features[t, base:base+PLAYER_FEAT_DIM] = [
                        px, py, pvx, pvy, dists_sorted[i], team_proxy[i]]

        if verbose:
            print(f"\n  Completado: {T} frames  shape={features.shape}")
        return features

    def extract_from_game_dir(
        self,
        game_dir:    Path,
        output_path: Optional[str] = None,
        verbose:     bool = True,
    ) -> np.ndarray:
        """
        Lee todos los frames .jpg de una carpeta de partido,
        los ordena por número de frame y extrae trayectorias.

        Nombre de fichero esperado: frameXXXXX.jpg
        """
        jpg_files = sorted(
            game_dir.glob("*.jpg"),
            key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0)
        )

        if len(jpg_files) == 0:
            print(f"  [WARN] Sin .jpg en {game_dir}")
            return np.zeros((0, TRAJ_FEAT_DIM), dtype=np.float32)

        if verbose:
            print(f"  {game_dir.name}  ({len(jpg_files)} frames)")

        frames = [cv2.imread(str(p)) for p in jpg_files]
        # Filtrar frames que no se pudieron leer
        frames = [f for f in frames if f is not None]

        features = self.extract_from_frames(frames, verbose=verbose)

        if output_path:
            np.save(output_path, features)
            if verbose:
                print(f"  Guardado: {output_path}")

        return features


# ─────────────────────────────────────────────────────────────────────────────
# BATCH EXTRACTION — itera por carpetas de partido
# ─────────────────────────────────────────────────────────────────────────────

def run_batch_extraction(
    frame_dir:         str,
    output_dir:        str,
    ball_model_path:   str   = "ball_detection_model.pt",
    player_model_path: str   = "yolov8m.pt",
    conf_ball:         float = 0.25,
    conf_player:       float = 0.40,
):
    """
    Recorre frame_dir buscando carpetas de partido (las que contienen .jpg),
    extrae trayectorias y guarda un .npy por partido en output_dir.

    Estructura esperada:
        frame_dir/
          liga/
            temporada/
              fecha - equipo1 - equipo2/
                frame00001.jpg
                ...
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_dir = Path(frame_dir)

    extractor = TrajectoryExtractor(
        ball_model_path=ball_model_path,
        player_model_path=player_model_path,
        conf_ball=conf_ball,
        conf_player=conf_player,
    )

    # Encontrar todas las carpetas que contienen .jpg directamente
    game_dirs = sorted([
        p for p in frame_dir.rglob("*")
        if p.is_dir() and any(p.glob("*.jpg"))
    ])

    print(f"\n[BatchExtraction] {len(game_dirs)} partidos encontrados en {frame_dir}\n")

    for i, game_dir in enumerate(game_dirs):
        key      = game_dir_to_key(game_dir, frame_dir)
        out_path = os.path.join(output_dir, key + ".npy")

        if os.path.exists(out_path):
            print(f"  [{i+1}/{len(game_dirs)}] Ya existe: {key}.npy, saltando.")
            continue

        print(f"  [{i+1}/{len(game_dirs)}] {key}")
        try:
            extractor.extract_from_game_dir(
                game_dir=game_dir,
                output_path=out_path,
                verbose=True,
            )
        except Exception as e:
            print(f"  ERROR en {game_dir.name}: {e}")

    print("\n[BatchExtraction] Completado.")


# ─────────────────────────────────────────────────────────────────────────────
# LOADER (usado en el training loop desde main_spotting_traj.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_trajectory_features(
    traj_dir:   str,
    video_name: str,
    clip_start: int,
    clip_len:   int,
) -> torch.Tensor:
    """
    Carga features de un clip concreto desde disco.
    video_name debe coincidir con la clave usada en run_batch_extraction
    (e.g. 'england_efl_2019-2020_2019-10-01 - Blackburn Rovers - Nottingham Forest')

    Devuelve Tensor (clip_len, 139).
    Si el archivo no existe devuelve ceros (no rompe el training).
    """
    path = os.path.join(traj_dir, f"{video_name}.npy")
    if not os.path.exists(path):
        return torch.zeros(clip_len, TRAJ_FEAT_DIM, dtype=torch.float32)

    all_feat = np.load(path)
    end      = clip_start + clip_len

    if end > len(all_feat):
        feat = all_feat[clip_start:]
        pad  = np.zeros((end - len(all_feat), TRAJ_FEAT_DIM), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=0)
    else:
        feat = all_feat[clip_start:end]

    return torch.from_numpy(feat.copy())


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULOS PYTORCH
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryFusionEncoder(nn.Module):
    """Proyecta traj features (139) al espacio del backbone y las fusiona."""
    def __init__(self, traj_dim=TRAJ_FEAT_DIM, visual_dim=256,
                 hidden_dim=128, fusion_mode="add", dropout=0.1):
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
        if fusion_mode == "concat":
            self.fusion_proj = nn.Linear(visual_dim * 2, visual_dim)

    def forward(self, visual_tokens, traj_features):
        traj_proj = self.projector(traj_features)
        if self.fusion_mode == "add":
            return visual_tokens + traj_proj
        return self.fusion_proj(torch.cat([visual_tokens, traj_proj], dim=-1))


class BallAttentionGate(nn.Module):
    """Gate temporal por velocidad del balón. Pondera más frames pre-acción."""
    def __init__(self, visual_dim=256, ball_feat_dim=BALL_FEAT_DIM):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(ball_feat_dim, visual_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(visual_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens, traj_features):
        attn = self.gate(traj_features[..., :BALL_FEAT_DIM])  # (B, T, 1)
        return tokens * attn


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def visualize_trajectories(
    game_dir:      str,
    traj_features: np.ndarray,
    output_path:   str = "viz_traj.mp4",
    fps:           float = 25.0,
    max_frames:    int = 500,
):
    """
    Renderiza trayectorias sobre los frames de un partido.
    game_dir: carpeta con los .jpg del partido.
    """
    game_dir   = Path(game_dir)
    jpg_files  = sorted(
        game_dir.glob("*.jpg"),
        key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0)
    )[:max_frames]

    if not jpg_files:
        print(f"[Visualización] Sin .jpg en {game_dir}")
        return

    sample = cv2.imread(str(jpg_files[0]))
    H, W   = sample.shape[:2]
    out    = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    ball_trail = []

    for t, jpg in enumerate(jpg_files):
        if t >= len(traj_features):
            break
        frame = cv2.imread(str(jpg))
        if frame is None:
            continue

        feat                             = traj_features[t]
        bx, by, vx, vy, ax, ay, visible = feat[:BALL_FEAT_DIM]

        if visible > 0.5:
            bx_px, by_px = int(bx * W), int(by * H)
            ball_trail.append((bx_px, by_px))
            if len(ball_trail) > 20:
                ball_trail.pop(0)
            for k in range(1, len(ball_trail)):
                alpha = k / len(ball_trail)
                cv2.line(frame, ball_trail[k-1], ball_trail[k],
                         (int(255*alpha), int(150*alpha), 0), 2)
            cv2.circle(frame, (bx_px, by_px), 10, (0, 220, 255), -1)
            speed = np.sqrt(vx**2 + vy**2)
            if speed > 0.002:
                cv2.arrowedLine(frame, (bx_px, by_px),
                                (int(bx_px+vx*W*80), int(by_px+vy*H*80)),
                                (0, 255, 255), 2)

        for i in range(MAX_PLAYERS):
            base = BALL_FEAT_DIM + i * PLAYER_FEAT_DIM
            px, py = feat[base], feat[base+1]
            if px == 0 and py == 0:
                continue
            team  = feat[base+5]
            color = (0, 210, 0) if team > 0.5 else (0, 0, 210)
            cv2.circle(frame, (int(px*W), int(py*H)), 5, color, -1)

        cv2.putText(frame,
                    f"Frame {t:05d} | ball_speed={np.sqrt(vx**2+vy**2)*W:.1f}px/f | vis={visible:.0f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        out.write(frame)

    out.release()
    print(f"[Visualización] Guardado en {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",          choices=["extract","visualize","info"], default="extract")
    # extract
    p.add_argument("--frame_dir",     type=str, default=None,
                   help="Raíz de frames (con subcarpetas por partido)")
    p.add_argument("--output_dir",    type=str, default="./traj_features")
    p.add_argument("--ball_model",    type=str, default="ball_detection_model.pt")
    p.add_argument("--player_model",  type=str, default="yolov8m.pt")
    p.add_argument("--conf_ball",     type=float, default=0.25)
    p.add_argument("--conf_player",   type=float, default=0.40)
    # visualize
    p.add_argument("--game_dir",      type=str, default=None,
                   help="Carpeta de un partido con .jpg (modo visualize)")
    p.add_argument("--traj_npy",      type=str, default=None)
    p.add_argument("--visual_output", type=str, default="viz_traj.mp4")
    p.add_argument("--fps",           type=float, default=25.0)
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "info":
        print("=" * 55)
        print("  DUAL-MODEL TRAJECTORY PIPELINE")
        print("=" * 55)
        print(f"  TRAJ_FEAT_DIM   = {TRAJ_FEAT_DIM}")
        print(f"  BALL_FEAT_DIM   = {BALL_FEAT_DIM}  [x,y,vx,vy,ax,ay,visible]")
        print(f"  PLAYER_FEAT_DIM = {PLAYER_FEAT_DIM}  [x,y,vx,vy,dist_ball,team]")
        print(f"  MAX_PLAYERS     = {MAX_PLAYERS}")
        print(f"  Balón    → ball_detection_model.pt  (fine-tuned, clase 0)")
        print(f"  Jugadores→ yolov8m.pt               (COCO, clase 0 = person)")

    elif args.mode == "extract":
        if not args.frame_dir:
            print("ERROR: --frame_dir requerido"); return
        run_batch_extraction(
            frame_dir=args.frame_dir,
            output_dir=args.output_dir,
            ball_model_path=args.ball_model,
            player_model_path=args.player_model,
            conf_ball=args.conf_ball,
            conf_player=args.conf_player,
        )

    elif args.mode == "visualize":
        if not args.game_dir or not args.traj_npy:
            print("ERROR: --game_dir y --traj_npy requeridos"); return
        feat = np.load(args.traj_npy)
        print(f"Features: shape={feat.shape}")
        visualize_trajectories(
            game_dir=args.game_dir,
            traj_features=feat,
            output_path=args.visual_output,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()