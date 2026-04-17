#!/usr/bin/env python3
"""
main_spotting_traj.py
=====================
Script de entrenamiento modificado para usar las features de trayectoria
con los modelos model_bigru_traj / model_spotting_traj.

Cambios respecto al main_spotting.py original:
  1. TrajDatasetWrapper: envuelve vuestro dataset existente y añade
     la clave 'traj' en cada batch cargando el .npy pre-extraído.
  2. Opción --extract_traj para ejecutar la extracción antes de entrenar.
  3. Parámetros nuevos en el config JSON:
       "use_traj"   : true / false
       "traj_dir"   : "/path/to/traj_features"
       "traj_fusion": "add" / "concat"

Uso:
    # 1. Extraer trayectorias (una sola vez, offline)
    python trajectory_pipeline.py --mode extract \
        --video_dir /data/videos --output_dir /data/traj_features \
        --model_path yolov8m.pt

    # 2. Entrenar con trayectorias
    python main_spotting_traj.py --model my_traj_config --gpu 0

    # 3. Entrenar SIN trayectorias (ablación, mismos pesos)
    python main_spotting_traj.py --model my_baseline_config --gpu 0

Ejemplo de config JSON (config/my_traj_config.json):
    {
      "model_module"   : "model_bigru_traj",
      "use_traj"       : true,
      "traj_dir"       : "/data/traj_features",
      "traj_fusion"    : "add",
      "frame_dir"      : "/data/frames",
      "save_dir"       : "/experiments",
      "labels_dir"     : "/data/labels",
      "store_mode"     : "load",
      "task"           : "spotting",
      "batch_size"     : 4,
      "clip_len"       : 100,
      "dataset"        : "SoccerNetBall",
      "epoch_num_frames": 200000,
      "feature_arch"   : "rny002",
      "learning_rate"  : 1e-4,
      "num_classes"    : 12,
      "num_epochs"     : 50,
      "warm_up_epochs" : 3,
      "only_test"      : false,
      "device"         : "cuda",
      "num_workers"    : 4
    }
"""

# ── Parse arguments ────────────────────────────────────────────────────────
import argparse
import os
import sys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed',  type=int, default=1)
    parser.add_argument('--gpu',   type=str, default='0')
    parser.add_argument('--extract_traj', action='store_true',
                        help='Ejecutar extracción de trayectorias antes de entrenar')
    parser.add_argument('--traj_model', type=str, default='yolov8m.pt',
                        help='Checkpoint YOLOv8 para extracción de trayectorias')
    return parser.parse_args()

args = get_args()

os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# ── Standard imports ───────────────────────────────────────────────────────
import torch
import numpy as np
import random
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tabulate import tabulate
import importlib
from pathlib import Path
from typing import Optional

# ── Local imports ──────────────────────────────────────────────────────────
from util.io import load_json, store_json
from util.eval_spotting import evaluate
from dataset.datasets import get_datasets

AP10_EXCLUDED = {'FREE KICK', 'GOAL'}


# ─────────────────────────────────────────────────────────────────────────────
# TRAJECTORY DATASET WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class TrajDatasetWrapper(Dataset):
    """
    Envuelve el dataset original (devuelto por get_datasets) y añade
    la clave 'traj' en cada sample, cargando las features pre-extraídas
    desde traj_dir.

    El archivo esperado es:
        traj_dir / <game_id>.npy   shape: (total_frames, 139)

    donde <game_id> es el identificador del vídeo que el dataset original
    expone en cada sample bajo la clave 'game_id' (o 'video_id').

    Si el archivo no existe para un vídeo, devuelve ceros (modo degradado)
    para no romper el entrenamiento.

    IMPORTANTE: adaptar _get_game_id() si vuestro dataset usa una clave
    diferente para identificar el vídeo/partido.
    """

    def __init__(self, base_dataset, traj_dir, clip_len):
        self.base     = base_dataset
        self.traj_dir = traj_dir
        self.clip_len = clip_len
        self._cache   = {}
        self.TRAJ_FEAT_DIM = 7 + 6 * 22

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        # Delega cualquier atributo desconocido al dataset base
        # Esto incluye 'videos', 'classes', y lo que sea que use evaluate()
        return getattr(self.base, name)

    def __getitem__(self, idx):
        sample     = self.base[idx]
        game_id    = self._get_game_id(sample)
        clip_start = self._get_clip_start(sample)
        traj       = self._load_traj(game_id, clip_start)
        sample['traj'] = traj
        return sample

    def _get_game_id(self, sample: dict) -> str:
        """
        Extrae el identificador del partido del sample.
        Prueba varias claves habituales; adaptar si es necesario.
        """
        for key in ('game_id', 'video_id', 'video_name', 'game', 'match_id'):
            if key in sample:
                return str(sample[key])
        # Si ninguna clave encaja, usar índice (último recurso)
        return 'unknown'

    def _get_clip_start(self, sample: dict) -> int:
        """
        Extrae el frame de inicio del clip del sample.
        """
        for key in ('clip_start', 'start_frame', 'frame_start', 'start'):
            if key in sample:
                return int(sample[key])
        return 0

    def _load_traj(self, game_id: str, clip_start: int) -> torch.Tensor:
        """
        Carga (o recupera de caché) las features de trayectoria para
        el clip [clip_start : clip_start + clip_len].
        """
        # Intentar varias extensiones de nombre de archivo
        candidates = [
            os.path.join(self.traj_dir, f'{game_id}.npy'),
            os.path.join(self.traj_dir, f'{game_id.replace("/", "_")}.npy'),
            os.path.join(self.traj_dir, f'{Path(game_id).stem}.npy'),
        ]

        npy_path = None
        for c in candidates:
            if os.path.exists(c):
                npy_path = c
                break

        if npy_path is None:
            # Archivo no encontrado: devolver ceros (modo degradado)
            return torch.zeros(self.clip_len, self.TRAJ_FEAT_DIM,
                               dtype=torch.float32)

        # Caché en memoria para evitar recargas repetidas del mismo vídeo
        if npy_path not in self._cache:
            self._cache[npy_path] = np.load(npy_path)  # (total_frames, 139)

        all_feat = self._cache[npy_path]
        end = clip_start + self.clip_len

        if end > len(all_feat):
            feat = all_feat[clip_start:]
            pad  = np.zeros((end - len(all_feat), self.TRAJ_FEAT_DIM),
                            dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=0)
        else:
            feat = all_feat[clip_start:end]

        return torch.from_numpy(feat.copy())


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (iguales que main_spotting.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mAP(ap_score, classes, exclude=None):
    excluded = set() if exclude is None else set(exclude)
    values = [
        ap_score[i]
        for i, class_name in enumerate(classes.keys())
        if class_name not in excluded
    ]
    return float(np.mean(values)) if values else 0.0


def update_args(args, config):
    args.frame_dir       = config['frame_dir']
    args.save_dir        = config['save_dir'] + '/' + args.model
    args.store_dir       = config['save_dir'] + '/' + 'splits'
    args.labels_dir      = config['labels_dir']
    args.store_mode      = config['store_mode']
    args.task            = config['task']
    args.batch_size      = config['batch_size']
    args.clip_len        = config['clip_len']
    args.dataset         = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch    = config['feature_arch']
    args.learning_rate   = config['learning_rate']
    args.num_classes     = config['num_classes']
    args.num_epochs      = config['num_epochs']
    args.warm_up_epochs  = config['warm_up_epochs']
    args.only_test       = config['only_test']
    args.device          = config['device']
    args.num_workers     = config['num_workers']

    # ── [NUEVO] parámetros de trayectorias ──────────────────────────────
    args.use_traj    = config.get('use_traj', False)
    args.traj_dir    = config.get('traj_dir', './traj_features')
    args.traj_fusion = config.get('traj_fusion', 'add')

    return args


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── GPU info ───────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU detected.')
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f'Using GPU: {gpu_name}  (CUDA_VISIBLE_DEVICES={args.gpu})')

    # ── Seed ───────────────────────────────────────────────────────────────
    print('Setting seed:', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── Config ─────────────────────────────────────────────────────────────
    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args   = update_args(args, config)

    print(f'\n{"="*55}')
    print(f'  Trayectorias: {"ACTIVAS ✓" if args.use_traj else "INACTIVAS (baseline)"}')
    if args.use_traj:
        print(f'  traj_dir   : {args.traj_dir}')
        print(f'  fusion_mode: {args.traj_fusion}')
    print(f'{"="*55}\n')

    # ── Extracción de trayectorias (opcional) ──────────────────────────────
    if args.extract_traj and args.use_traj:
        print('[Main] Extrayendo trayectorias (esto puede tardar)...')
        from trajectory_pipeline import run_batch_extraction
        run_batch_extraction(
            video_dir=args.frame_dir,   # adaptar si los vídeos están en otro path
            output_dir=args.traj_dir,
            model_path=args.traj_model,
        )
        print('[Main] Extracción completada.\n')

    # ── Importar modelo según config ───────────────────────────────────────
    Model = importlib.import_module(f"model.{config['model_module']}").Model

    # ── Directorios ────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Datasets (igual que antes) ─────────────────────────────────────────
    classes, train_data, val_data, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets stored. Re-run with store_mode="load".')
        sys.exit(0)
    print('Datasets loaded correctly.')

    # ── [NUEVO] Envolver datasets con TrajDatasetWrapper ──────────────────
    if args.use_traj:
        print(f'[Main] Envolviendo datasets con TrajDatasetWrapper '
              f'(traj_dir={args.traj_dir})')
        train_data = TrajDatasetWrapper(train_data, args.traj_dir, args.clip_len)
        val_data   = TrajDatasetWrapper(val_data,   args.traj_dir, args.clip_len)
        test_data  = TrajDatasetWrapper(test_data,  args.traj_dir, args.clip_len)

    # ── Modelo ─────────────────────────────────────────────────────────────
    model = Model(args=args)

    if not args.only_test:
        optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

        def worker_init_fn(id):
            random.seed(id + epoch * 100)

        train_loader = DataLoader(
            train_data, shuffle=False, batch_size=args.batch_size,
            pin_memory=True, num_workers=args.num_workers,
            prefetch_factor=(2 if args.num_workers > 0 else None),
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_data, shuffle=False, batch_size=args.batch_size,
            pin_memory=True, num_workers=args.num_workers,
            prefetch_factor=(2 if args.num_workers > 0 else None),
            worker_init_fn=worker_init_fn,
        )

        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch)

        losses       = []
        best_criterion = float('inf')
        epoch        = 0

        print('START TRAINING')
        for epoch in range(num_epochs):
            train_loss = model.epoch(
                train_loader, optimizer, scaler, lr_scheduler=lr_scheduler)
            val_loss   = model.epoch(val_loader)

            better = val_loss < best_criterion
            if better:
                best_criterion = val_loss

            print('[Epoch {}] Train: {:.5f}  Val: {:.5f}{}'.format(
                epoch, train_loss, val_loss,
                '  ← mejor!' if better else ''))

            losses.append({'epoch': epoch, 'train': train_loss, 'val': val_loss})
            store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)

            if better:
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, 'checkpoint_best.pt'))

    # ── Inferencia ─────────────────────────────────────────────────────────
    print('\nSTART INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))

    # Parámetros y MACs
    num_params = sum(p.numel() for p in model._model.parameters())
    try:
        from thop import profile
        dummy_input = torch.zeros(1, args.clip_len, 3, 224, 398, device=model.device)
        macs, _ = profile(model._model, inputs=(dummy_input,), verbose=False)
        macs_str = f'{macs/1e9:.2f} GMACs'
    except Exception:
        macs_str = 'N/A'
    print(f'Model params: {num_params:,}  |  MACs: {macs_str}')

    # Evaluación
    map12, ap_score = evaluate(model, test_data, nms_window=5)
    map10 = compute_mAP(ap_score, classes, exclude=AP10_EXCLUDED)

    # Tabla de resultados
    table = [
        [cn + (' (*)' if cn in AP10_EXCLUDED else ''), f'{ap_score[i]*100:.2f}']
        for i, cn in enumerate(classes.keys())
    ]
    print(tabulate(table, ['Class', 'AP'], tablefmt='grid'))
    print(tabulate([
        ['mAP12 (all)',            f'{map12*100:.2f}'],
        ['mAP10 (excl. FK & GK)', f'{map10*100:.2f}'],
    ], ['Metric', 'AP'], tablefmt='grid'))
    print('(*) excluded from mAP10')
    print('DONE')


if __name__ == '__main__':
    main(args)
