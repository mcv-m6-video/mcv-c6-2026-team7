#!/usr/bin/env python3
"""
compare_models.py — Multi-model discovery, inference and comparison.

Steps
─────
1. Scans config/ for every *.json and checks whether a trained checkpoint
   exists for each one.  Prints a status table to the terminal.

2. Runs inference (on the VAL split by default, pass --split test to use
   the test split) for every model that has a checkpoint.

3. Saves per-model results to  results/comparison/results.json
   and produces four comparison plots:
       • bar_ap_per_class.png   — per-class AP grouped by model
       • bar_map12_map10.png    — AP12 / AP10 side-by-side per model
       • heatmap_ap.png         — models × classes heatmap
       • radar_ap.png           — radar / spider chart per model

Usage
─────
    python compare_models.py                      # val split, best_ap ckpt
    python compare_models.py --split test
    python compare_models.py --ckpt best          # use checkpoint_best.pt
    python compare_models.py --models baseline convnext   # subset only
"""

import argparse
import importlib
import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tabulate import tabulate

# ── project-local imports ─────────────────────────────────────────────────────
from util.io import load_json, store_json
from util.eval_classification import evaluate, compute_mAP, AP10_EXCLUDED
from dataset.datasets import get_datasets

# ─────────────────────────────────────────────────────────────────────────────
TICK  = '✔'
CROSS = '✘'
CONFIG_DIR  = 'config'
OUTPUT_DIR  = os.path.join('results', 'comparison')
CKPT_BEST_AP = 'checkpoint_best_ap.pt'
CKPT_BEST    = 'checkpoint_best.pt'
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description='Compare multiple trained models.')
    p.add_argument('--split',  default='val', choices=['val', 'test'],
                   help='Dataset split to run inference on (default: val).')
    p.add_argument('--ckpt',   default='best_ap',
                   choices=['best_ap', 'best'],
                   help='Which checkpoint to load: best_ap (mAP-based) '
                        'or best (loss-based). Default: best_ap.')
    p.add_argument('--models', nargs='*', default=None,
                   help='Optional whitelist of model names (without .json). '
                        'If omitted all configs are used.')
    p.add_argument('--seed',   type=int, default=1)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Discovery
# ══════════════════════════════════════════════════════════════════════════════

def discover_models(whitelist=None, ckpt_pref='best_ap'):
    """
    Scan config/ for *.json files (excluding README).
    For each, check whether the corresponding checkpoint exists.

    Returns a list of dicts:
        { name, config_path, config, ckpt_path, has_ckpt }
    """
    cfg_files = sorted(
        f for f in os.listdir(CONFIG_DIR)
        if f.endswith('.json') and f.lower() != 'readme.md'
    )

    models = []
    for cfg_file in cfg_files:
        name = cfg_file[:-5]          # strip .json
        if whitelist and name not in whitelist:
            continue

        config_path = os.path.join(CONFIG_DIR, cfg_file)
        try:
            config = load_json(config_path)
        except Exception as e:
            print(f'  [WARN] Could not parse {cfg_file}: {e}')
            continue

        save_dir = config.get('save_dir', 'results') + '/' + name
        ckpt_dir = os.path.join(save_dir, 'checkpoints')

        # Preferred checkpoint, fall back to the other one
        if ckpt_pref == 'best_ap':
            preferred = os.path.join(ckpt_dir, CKPT_BEST_AP)
            fallback  = os.path.join(ckpt_dir, CKPT_BEST)
        else:
            preferred = os.path.join(ckpt_dir, CKPT_BEST)
            fallback  = os.path.join(ckpt_dir, CKPT_BEST_AP)

        if os.path.exists(preferred):
            ckpt_path = preferred
            has_ckpt  = True
        elif os.path.exists(fallback):
            ckpt_path = fallback
            has_ckpt  = True
        else:
            ckpt_path = None
            has_ckpt  = False

        models.append({
            'name'       : name,
            'config_path': config_path,
            'config'     : config,
            'ckpt_path'  : ckpt_path,
            'has_ckpt'   : has_ckpt,
        })

    return models


def print_discovery_table(models):
    """Pretty-print the discovery status to the terminal."""
    rows = []
    for m in models:
        cfg_status  = f'{TICK}  {m["name"]}.json'
        ckpt_label  = os.path.basename(m['ckpt_path']) if m['ckpt_path'] else '—'
        ckpt_status = f'{TICK}  {ckpt_label}' if m['has_ckpt'] \
                      else f'{CROSS}  no checkpoint found'
        rows.append([cfg_status, ckpt_status])

    print('\n' + '═' * 70)
    print('  MODEL DISCOVERY')
    print('═' * 70)
    print(tabulate(rows,
                   headers=['Config', 'Checkpoint'],
                   tablefmt='simple'))
    ready = sum(1 for m in models if m['has_ckpt'])
    print(f'\n  {ready}/{len(models)} models ready for inference.')
    print('═' * 70 + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Inference
# ══════════════════════════════════════════════════════════════════════════════

def update_args_from_config(args, config):
    """Mirror of update_args in main_classification.py."""
    args.frame_dir          = config['frame_dir']
    args.save_dir           = config['save_dir'] + '/' + args._model_name
    args.store_dir          = config['save_dir'] + '/' + 'splits'
    args.labels_dir         = config['labels_dir']
    args.store_mode         = config['store_mode']
    args.task               = config['task']
    args.batch_size         = config['batch_size']
    args.clip_len           = config['clip_len']
    args.dataset            = config['dataset']
    args.epoch_num_frames   = config['epoch_num_frames']
    args.feature_arch       = config['feature_arch']
    args.learning_rate      = config['learning_rate']
    args.num_classes        = config['num_classes']
    args.num_epochs         = config['num_epochs']
    args.warm_up_epochs     = config['warm_up_epochs']
    args.only_test          = config['only_test']
    args.device             = config['device']
    args.num_workers        = config['num_workers']
    args.loss_type          = config.get('loss_type', 'bce')
    args.transformer_dropout= config.get('transformer_dropout', 0.1)
    args.transformer_layers = config.get('transformer_layers', 2)
    args.teacher_model      = config.get('teacher_model', None)
    args.distill_alpha      = config.get('distill_alpha', 0.5)
    args.distill_temp       = config.get('distill_temp', 4.0)
    return args


def run_inference(model_info, args_template, split='val'):
    """
    Load a model from its checkpoint and run evaluate() on the chosen split.
    Returns { ap_per_class (ndarray), map12, map10 } or None on failure.
    """
    import torch
    import argparse as _ap

    name   = model_info['name']
    config = model_info['config']

    # Build a fresh args namespace for this model
    m_args             = _ap.Namespace(**vars(args_template))
    m_args._model_name = name
    m_args             = update_args_from_config(m_args, config)
    m_args.store_mode  = 'load'   # never re-store during comparison

    print(f'\n── Inference: {name} ──')

    try:
        classes, train_data, val_data, test_data = get_datasets(m_args)
        dataset = val_data if split == 'val' else test_data

        Model = importlib.import_module(
            f"model.{config['model_module']}").Model
        model = Model(args=m_args)

        import torch
        ckpt = torch.load(model_info['ckpt_path'],
                          map_location=getattr(model, 'device', 'cpu'))
        model.load(ckpt)

        ap = evaluate(model, dataset)        # (num_classes,)
        map12 = compute_mAP(ap, classes)
        map10 = compute_mAP(ap, classes, exclude=AP10_EXCLUDED)

        print(f'  AP12={map12*100:.2f}%  AP10={map10*100:.2f}%')
        return {
            'ap_per_class': ap,
            'map12'       : map12,
            'map10'       : map10,
            'classes'     : classes,
        }

    except Exception as e:
        import traceback
        print(f'  [ERROR] {name}: {e}')
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Plots
# ══════════════════════════════════════════════════════════════════════════════

# A colourblind-friendly palette that scales gracefully
_PALETTE = [
    '#4878CF', '#D65F5F', '#6ACC65', '#B47CC7',
    '#C4AD66', '#77BEDB', '#E87F03', '#A7414A',
    '#565656', '#0F9B8E', '#9E4E6A', '#3E9651',
]

def _colours(n):
    if n <= len(_PALETTE):
        return _PALETTE[:n]
    return [cm.tab20(i / n) for i in range(n)]


def plot_bar_per_class(results, class_names, save_dir):
    """
    Grouped bar chart: x = class, groups = models.
    Saves bar_ap_per_class.png
    """
    model_names = list(results.keys())
    n_models    = len(model_names)
    n_classes   = len(class_names)
    colours     = _colours(n_models)

    x      = np.arange(n_classes)
    width  = 0.8 / n_models
    offset = np.linspace(-(0.8 - width) / 2, (0.8 - width) / 2, n_models)

    fig, ax = plt.subplots(figsize=(max(14, n_classes * 1.2), 5),
                           constrained_layout=True)

    for i, (mname, res) in enumerate(results.items()):
        ap = res['ap_per_class'] * 100
        ax.bar(x + offset[i], ap, width, label=mname,
               color=colours[i], edgecolor='none', alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('AP (%)', fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title('Per-class AP — model comparison', fontsize=12)
    ax.legend(fontsize=8, ncol=max(1, n_models // 4))
    ax.grid(True, axis='y', linewidth=0.4, alpha=0.5)

    # Mark AP10-excluded classes
    for j, cls in enumerate(class_names):
        if cls in AP10_EXCLUDED:
            ax.axvspan(j - 0.45, j + 0.45, color='gray', alpha=0.08, zorder=0)
            ax.text(j, 1, '(*)', ha='center', va='bottom',
                    fontsize=6, color='gray', transform=ax.get_xaxis_transform())

    p = os.path.join(save_dir, 'bar_ap_per_class.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {p}')


def plot_bar_map_summary(results, save_dir):
    """
    Side-by-side AP12 / AP10 bars per model.
    Saves bar_map12_map10.png
    """
    model_names = list(results.keys())
    n_models    = len(model_names)
    colours     = _colours(n_models)

    map12_vals = [results[m]['map12'] * 100 for m in model_names]
    map10_vals = [results[m]['map10'] * 100 for m in model_names]

    x     = np.arange(n_models)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, n_models * 1.4), 5),
                           constrained_layout=True)

    bars12 = ax.bar(x - width / 2, map12_vals, width,
                    label='AP12 (all classes)', color=colours,
                    edgecolor='none', alpha=0.9)
    bars10 = ax.bar(x + width / 2, map10_vals, width,
                    label='AP10 (excl. FREE KICK & GOAL)',
                    color=colours, edgecolor='none', alpha=0.55,
                    hatch='//')

    # Value labels on top of bars
    for bar in list(bars12) + list(bars10):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                f'{h:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('mAP (%)', fontsize=10)
    ax.set_ylim(0, min(100, max(map12_vals + map10_vals) * 1.18))
    ax.set_title('AP12 and AP10 — model comparison', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linewidth=0.4, alpha=0.5)

    p = os.path.join(save_dir, 'bar_map12_map10.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {p}')


def plot_heatmap(results, class_names, save_dir):
    """
    Heatmap: rows = models, columns = classes.
    Saves heatmap_ap.png
    """
    model_names = list(results.keys())
    n_models    = len(model_names)
    n_classes   = len(class_names)

    mat = np.array([results[m]['ap_per_class'] * 100
                    for m in model_names])          # (M, C)

    fig, ax = plt.subplots(
        figsize=(max(10, n_classes * 0.85), max(3, n_models * 0.7)),
        constrained_layout=True)

    im = ax.imshow(mat, aspect='auto', cmap='YlGnBu', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label='AP (%)', fraction=0.03, pad=0.02)

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title('AP heatmap — models × classes', fontsize=12)

    # Cell annotations
    for r in range(n_models):
        for c in range(n_classes):
            ax.text(c, r, f'{mat[r, c]:.1f}',
                    ha='center', va='center', fontsize=6,
                    color='black' if mat[r, c] < 60 else 'white')

    # Shade AP10-excluded columns
    for j, cls in enumerate(class_names):
        if cls in AP10_EXCLUDED:
            ax.axvspan(j - 0.5, j + 0.5, color='red', alpha=0.08, zorder=0)

    p = os.path.join(save_dir, 'heatmap_ap.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {p}')


def plot_radar(results, class_names, save_dir):
    """
    Radar / spider chart: one polygon per model, axes = classes.
    Saves radar_ap.png
    """
    model_names = list(results.keys())
    colours     = _colours(len(model_names))
    n_classes   = len(class_names)

    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False).tolist()
    angles += angles[:1]          # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True),
                           constrained_layout=True)

    for i, (mname, res) in enumerate(results.items()):
        vals = (res['ap_per_class'] * 100).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.6, color=colours[i], label=mname)
        ax.fill(angles, vals, color=colours[i], alpha=0.10)

    ax.set_thetagrids(np.degrees(angles[:-1]), class_names, fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_title('Per-class AP radar — model comparison',
                 fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    ax.grid(True, linewidth=0.5, alpha=0.6)

    p = os.path.join(save_dir, 'radar_ap.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {p}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse as _ap
    args = get_args()

    # ── Step 1: discover ─────────────────────────────────────────────────────
    models = discover_models(whitelist=args.models, ckpt_pref=args.ckpt)

    if not models:
        sys.exit('No configs found in config/. '
                 'Run from the Week5 root directory.')

    print_discovery_table(models)

    ready = [m for m in models if m['has_ckpt']]
    if not ready:
        sys.exit('No models have checkpoints — nothing to evaluate.')

    # ── Step 2: inference ────────────────────────────────────────────────────
    # Build a minimal args namespace as template (no model-specific fields yet)
    args_template      = _ap.Namespace()
    args_template.seed = args.seed
    args_template.only_test = False    # needed by some model constructors

    results     = {}   # model_name → { ap_per_class, map12, map10, classes }
    class_names = None

    for m in ready:
        res = run_inference(m, args_template, split=args.split)
        if res is None:
            continue
        results[m['name']] = res
        if class_names is None:
            class_names = list(res['classes'].keys())

    if not results:
        sys.exit('All inference runs failed.')

    # ── Print summary table ───────────────────────────────────────────────────
    print('\n' + '═' * 70)
    print('  RESULTS SUMMARY')
    print('═' * 70)
    header = ['Model'] + class_names + ['AP12', 'AP10']
    rows   = []
    for mname, res in results.items():
        row = [mname]
        row += [f"{v*100:.1f}" for v in res['ap_per_class']]
        row += [f"{res['map12']*100:.2f}", f"{res['map10']*100:.2f}"]
        rows.append(row)
    print(tabulate(rows, headers=header, tablefmt='simple'))
    print('═' * 70 + '\n')

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    serialisable = {}
    for mname, res in results.items():
        serialisable[mname] = {
            'map12'       : res['map12'],
            'map10'       : res['map10'],
            'ap_per_class': {cls: float(res['ap_per_class'][i])
                             for i, cls in enumerate(class_names)},
        }
    store_json(os.path.join(OUTPUT_DIR, 'results.json'),
               serialisable, pretty=True)
    print(f'  Results JSON saved → {os.path.join(OUTPUT_DIR, "results.json")}')

    # ── Step 3: plots ─────────────────────────────────────────────────────────
    print('\nGenerating comparison plots...')
    plot_bar_per_class(results, class_names, OUTPUT_DIR)
    plot_bar_map_summary(results, OUTPUT_DIR)
    plot_heatmap(results, class_names, OUTPUT_DIR)
    plot_radar(results, class_names, OUTPUT_DIR)

    print(f'\nAll outputs saved to  {OUTPUT_DIR}/')
    print('DONE')


if __name__ == '__main__':
    main()
