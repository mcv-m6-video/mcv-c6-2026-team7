#!/usr/bin/env python3
"""
File containing the main training script for T-DEED.
"""

#Standard imports
import argparse
import torch
import os
import numpy as np
import random
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
import sys
from torch.utils.data import DataLoader
from tabulate import tabulate
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe on clusters with no display
import matplotlib.pyplot as plt

#Local imports
import importlib
from util.io import load_json, store_json
from util.eval_classification import evaluate, compute_mAP, AP10_EXCLUDED
from util.eval_progress import (
    compute_class_frequencies, compute_weighted_ap_change
)
from dataset.datasets import get_datasets


# ── How often to run the AP eval (every X% of total epochs) ──────────────────
EVAL_EVERY_PCT   = 0.10   # 10 %
# Weighted-change early-stopping knobs
ALPHA            = 0.5    # 1.0 = only AP-based weights, 0.0 = only freq-based
PATIENCE         = 3      # stop after this many consecutive non-improving evals
MIN_IMPROVEMENT  = 0.0    # weighted_change must exceed this to count as progress
# ─────────────────────────────────────────────────────────────────────────────


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str, required=True)
    parser.add_argument('--seed',       type=int, default=1)
    parser.add_argument('--no_verbose', action='store_true',
                        help='Suppress all tables printed to stdout '
                             '(everything is still saved to disk).')
    parser.add_argument('--cfm',        action='store_true',
                        help='Compute and save confusion-matrix plots '
                             'at the end of inference.')
    return parser.parse_args()


def update_args(args, config):
    args.frame_dir       = config['frame_dir']
    args.save_dir        = config['save_dir'] + '/' + args.model
    args.store_dir       = config['save_dir'] + '/' + "splits"
    args.labels_dir      = config['labels_dir']
    args.store_mode      = config['store_mode']
    args.task            = config['task']
    args.batch_size      = config['batch_size']
    args.clip_len        = config['clip_len']
    args.dataset         = config['dataset']
    args.epoch_num_frames= config['epoch_num_frames']
    args.feature_arch    = config['feature_arch']
    args.learning_rate   = config['learning_rate']
    args.num_classes     = config['num_classes']
    args.num_epochs      = config['num_epochs']
    args.warm_up_epochs  = config['warm_up_epochs']
    args.only_test       = config['only_test']
    args.device          = config['device']
    args.num_workers         = config['num_workers']
    args.loss_type           = config.get('loss_type', 'bce')
    args.transformer_dropout = config.get('transformer_dropout', 0.1)
    args.transformer_layers  = config.get('transformer_layers', 2)
    args.teacher_model       = config.get('teacher_model', None)
    args.distill_alpha       = config.get('distill_alpha', 0.5)
    args.distill_temp        = config.get('distill_temp', 4.0)
    return args


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs)
    ])


def run_ap_eval(model, val_data, class_frequencies, prev_ap, classes,
                alpha=ALPHA, verbose=True):
    """
    Run evaluate(), compute the weighted-change metric and mAP summaries.
    Returns (current_ap, progress_dict).  prev_ap may be None on the first call.

    progress_dict keys:
        current_ap, per_class_change, weights,
        weighted_change, mean_ap, delta_mean_ap,
        map12, map10, delta_map12, delta_map10
    """
    current_ap = evaluate(model, val_data)          # np.ndarray (num_classes,)

    map12 = compute_mAP(current_ap, classes)
    map10 = compute_mAP(current_ap, classes, exclude=AP10_EXCLUDED)

    if prev_ap is None:
        progress = {
            'weighted_change' : None,
            'per_class_change': np.zeros_like(current_ap),
            'weights'         : np.ones_like(current_ap) / len(current_ap),
            'mean_ap'         : float(np.mean(current_ap)),
            'delta_mean_ap'   : None,
            'map12'           : map12,
            'map10'           : map10,
            'delta_map12'     : None,
            'delta_map10'     : None,
        }
    else:
        progress = compute_weighted_ap_change(
            current_ap, prev_ap, class_frequencies, alpha=alpha
        )
        prev_map12 = compute_mAP(prev_ap, classes)
        prev_map10 = compute_mAP(prev_ap, classes, exclude=AP10_EXCLUDED)
        progress['map12']      = map12
        progress['map10']      = map10
        progress['delta_map12'] = map12 - prev_map12
        progress['delta_map10'] = map10 - prev_map10

    progress['current_ap'] = current_ap

    if verbose:
        # ── per-class table ───────────────────────────────────────────────────
        table = []
        for i, cls in enumerate(classes.keys()):
            excluded = ' (*)' if cls in AP10_EXCLUDED else ''
            delta    = progress['per_class_change'][i]
            w        = progress['weights'][i]
            table.append([
                cls + excluded,
                f"{current_ap[i]*100:.2f}",
                f"{delta*100:+.2f}" if prev_ap is not None else "—",
                f"{w:.4f}",
            ])
        print(tabulate(table,
                       headers=["Class", "AP (%)", "ΔAP (%)", "Weight"],
                       tablefmt="grid"))
        print("(*) excluded from AP10")

        # ── mAP summary table ─────────────────────────────────────────────────
        def _fmt_delta(v):
            return f"{v*100:+.2f}%" if v is not None else "—"

        wc = progress['weighted_change']
        summary = [
            ["AP12 (all classes)",
             f"{map12*100:.2f}%",
             _fmt_delta(progress['delta_map12'])],
            ["AP10 (excl. FREE KICK & GOAL)",
             f"{map10*100:.2f}%",
             _fmt_delta(progress['delta_map10'])],
            ["Weighted Δ (early-stop metric)",
             f"{wc:+.5f}" if wc is not None else "—",
             ""],
        ]
        print(tabulate(summary,
                       headers=["Metric", "Value", "Δ vs prev eval"],
                       tablefmt="grid"))

    return current_ap, progress


def main(args):
    # ── Seed ──────────────────────────────────────────────────────────────────
    print('Setting seed to:', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config     = load_json('config/' + args.model + '.json')
    args       = update_args(args, config)
    Model      = importlib.import_module(f"model.{config['model_module']}").Model

    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    classes, train_data, val_data, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets stored. Re-run with load mode.')
        sys.exit()
    else:
        print('Datasets loaded correctly.')

    # Pre-compute class frequencies once from the training set
    print('Computing class frequencies from training set...')
    class_frequencies = compute_class_frequencies(train_data, args.num_classes)

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    model = Model(args=args)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    if not args.only_test:
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)

        # How many epochs between AP evals
        eval_every = max(1, int(round(num_epochs * EVAL_EVERY_PCT)))
        print(f'AP eval every {eval_every} epochs ({EVAL_EVERY_PCT*100:.0f}% of {num_epochs})')

        losses        = []
        ap_history    = []          # list of dicts, one per eval step
        best_criterion= float('inf')
        prev_ap       = None        # AP array from the previous eval step
        no_improve    = 0           # consecutive eval steps without progress
        epoch         = 0

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):

            train_loss = model.epoch(
                train_loader, optimizer, scaler, lr_scheduler=lr_scheduler)
            val_loss   = model.epoch(val_loader)

            # ── Standard loss-based best checkpoint ──────────────────────────
            better = val_loss < best_criterion
            if better:
                best_criterion = val_loss
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, 'checkpoint_best.pt'))

            print('[Epoch {}] Train loss: {:.5f}  Val loss: {:.5f}{}'.format(
                epoch, train_loss, val_loss, '  ← best' if better else ''))

            losses.append({'epoch': epoch, 'train': train_loss, 'val': val_loss})
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)

            # ── Periodic AP eval ─────────────────────────────────────────────
            is_last_epoch = (epoch == num_epochs - 1)
            if (epoch + 1) % eval_every == 0 or is_last_epoch:
                print(f'\n── AP Eval at epoch {epoch} ──')
                current_ap, progress = run_ap_eval(
                    model, val_data, class_frequencies, prev_ap, classes,
                    verbose=not args.no_verbose
                )

                # Save best AP checkpoint separately
                if prev_ap is None or progress['mean_ap'] > max(
                        (h['mean_ap'] for h in ap_history), default=0):
                    torch.save(model.state_dict(),
                               os.path.join(ckpt_dir, 'checkpoint_best_ap.pt'))
                    print('  → New best mAP checkpoint saved.')

                # Early stopping on weighted change
                wc = progress['weighted_change']
                if wc is not None:
                    if wc <= MIN_IMPROVEMENT:
                        no_improve += 1
                        print(f'  No AP progress ({no_improve}/{PATIENCE})')
                        if no_improve >= PATIENCE:
                            print('Early stopping triggered by weighted AP change.')
                            ap_history.append({'epoch': epoch, **progress})
                            store_json(os.path.join(args.save_dir, 'ap_history.json'),
                                       _serialisable(ap_history), pretty=True)
                            break
                    else:
                        no_improve = 0

                ap_history.append({'epoch': epoch, **progress})
                store_json(os.path.join(args.save_dir, 'ap_history.json'),
                           _serialisable(ap_history), pretty=True)

                prev_ap = current_ap
                print()

    # ── Final inference ───────────────────────────────────────────────────────
    print('START INFERENCE')
    best_ckpt = os.path.join(ckpt_dir, 'checkpoint_best_ap.pt')
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(ckpt_dir, 'checkpoint_best.pt')
    model.load(torch.load(best_ckpt))

    # Model size and MACs
    num_params = sum(p.numel() for p in model._model.parameters())
    try:
        from thop import profile
        dummy_input = torch.zeros(
            1, args.clip_len, 3, 224, 398, device=model.device)
        macs, _ = profile(model._model, inputs=(dummy_input,), verbose=False)
        macs_str = f"{macs/1e9:.2f} GMACs"
    except Exception:
        macs_str = "N/A (install thop: pip install thop)"

    print(f'\nModel params: {num_params:,}  |  MACs: {macs_str}')

    # Run inference — collect scores+labels for both AP and confusion matrix
    scores_all, labels_all = evaluate(model, test_data, return_raw=True)
    ap_score = _ap_from_raw(scores_all, labels_all)

    map12 = compute_mAP(ap_score, classes)
    map10 = compute_mAP(ap_score, classes, exclude=AP10_EXCLUDED)

    if not args.no_verbose:
        table = []
        for i, class_name in enumerate(classes.keys()):
            excluded = " (*)" if class_name in AP10_EXCLUDED else ""
            table.append([class_name + excluded, f"{ap_score[i]*100:.2f}"])
        print(tabulate(table, headers=["Class", "AP (%)"], tablefmt="grid"))

        summary_table = [
            ["AP12 (all classes)",            f"{map12*100:.2f}"],
            ["AP10 (excl. FREE KICK & GOAL)", f"{map10*100:.2f}"],
        ]
        print(tabulate(summary_table,
                       headers=["Metric", "Average Precision"],
                       tablefmt="grid"))
        print("(*) excluded from AP10")

    # Save final results to JSON
    final_results = {
        'map12'    : map12,
        'map10'    : map10,
        'per_class': {cls: float(ap_score[i])
                      for i, cls in enumerate(classes.keys())},
    }
    store_json(os.path.join(args.save_dir, 'final_results.json'),
               final_results, pretty=True)

    # ── Training-curve plots ──────────────────────────────────────────────────
    if ap_history:
        save_ap_plots(ap_history, classes, args.save_dir)

    # ── Confusion matrix ─────────────────────────────────────────────────────
    if args.cfm:
        save_confusion_matrices(scores_all, labels_all, classes, args.save_dir)

    print('CORRECTLY FINISHED TRAINING AND INFERENCE')


def _ap_from_raw(scores, labels):
    """Compute per-class AP from raw score/label arrays."""
    from sklearn.metrics import average_precision_score
    return average_precision_score(labels, scores, average=None)


def save_ap_plots(ap_history: list, classes: dict, save_dir: str) -> None:
    """
    Produce three figures from ap_history and save them to save_dir.

    Figure 1 — Absolute AP per class over eval steps.
    Figure 2 — ΔAP bars per class between consecutive evals.
    Figure 3 — AP12 and AP10 summary over eval steps (single plot).
    """
    import math

    class_names = list(classes.keys())
    num_classes  = len(class_names)
    epochs       = [entry['epoch'] for entry in ap_history]

    def _to_arr(v):
        return np.array(v.tolist() if isinstance(v, np.ndarray) else v)

    ap_matrix = np.array([_to_arr(e['current_ap'])        for e in ap_history])  # (T, C)
    deltas    = np.array([_to_arr(e['per_class_change'])   for e in ap_history])  # (T, C)
    map12_arr = np.array([e['map12'] for e in ap_history])                        # (T,)
    map10_arr = np.array([e['map10'] for e in ap_history])                        # (T,)

    ncols   = 4
    nrows   = math.ceil(num_classes / ncols)
    figsize = (ncols * 4, nrows * 3)

    def _hide_unused(axes_flat):
        for j in range(num_classes, len(axes_flat)):
            axes_flat[j].set_visible(False)

    # ── Figure 1: Absolute AP ─────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    fig1.suptitle('Per-class AP evolution (absolute)', fontsize=13)
    axes1_flat = axes1.flatten() if hasattr(axes1, 'flatten') else [axes1]

    for i, cls in enumerate(class_names):
        ax = axes1_flat[i]
        ax.plot(epochs, ap_matrix[:, i] * 100, marker='o', linewidth=1.5,
                markersize=4, color='steelblue')
        ax.set_ylim(0, 100)
        ax.set_title(cls, fontsize=9, pad=3)
        ax.set_xlabel('Epoch', fontsize=7)
        ax.set_ylabel('AP (%)', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    _hide_unused(axes1_flat)
    p = os.path.join(save_dir, 'ap_evolution.png')
    fig1.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f'  AP evolution plot saved → {p}')

    # ── Figure 2: ΔAP bars ───────────────────────────────────────────────────
    if len(ap_history) > 1:
        delta_epochs = epochs[1:]
        delta_matrix = deltas[1:]
        bar_width = (delta_epochs[1] - delta_epochs[0]) * 0.6 \
                    if len(delta_epochs) > 1 else 1

        fig2, axes2 = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
        fig2.suptitle('Per-class ΔAP between consecutive evals', fontsize=13)
        axes2_flat = axes2.flatten() if hasattr(axes2, 'flatten') else [axes2]

        for i, cls in enumerate(class_names):
            ax    = axes2_flat[i]
            vals  = delta_matrix[:, i] * 100
            colors = ['#2ecc71' if v > 0 else '#e74c3c' if v < 0 else 'gray'
                      for v in vals]
            ax.bar(delta_epochs, vals, color=colors, width=bar_width, edgecolor='none')
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_title(cls, fontsize=9, pad=3)
            ax.set_xlabel('Epoch', fontsize=7)
            ax.set_ylabel('ΔAP (%)', fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, axis='y', linewidth=0.4, alpha=0.5)

        _hide_unused(axes2_flat)
        p = os.path.join(save_dir, 'delta_ap_evolution.png')
        fig2.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f'  ΔAP evolution plot saved → {p}')
    else:
        print('  Not enough eval steps to plot ΔAP (need at least 2).')

    # ── Figure 3: AP12 & AP10 summary ────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax3.plot(epochs, map12_arr * 100, marker='o', linewidth=1.8,
             markersize=5, color='steelblue',  label='AP12 (all classes)')
    ax3.plot(epochs, map10_arr * 100, marker='s', linewidth=1.8,
             markersize=5, color='darkorange', label='AP10 (excl. FREE KICK & GOAL)',
             linestyle='--')
    ax3.set_ylim(0, 100)
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('mAP (%)', fontsize=10)
    ax3.set_title('AP12 and AP10 over training', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, linewidth=0.4, alpha=0.5)
    p = os.path.join(save_dir, 'map_summary_evolution.png')
    fig3.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f'  AP12/AP10 summary plot saved → {p}')


def save_confusion_matrices(scores: np.ndarray, labels: np.ndarray,
                             classes: dict, save_dir: str,
                             threshold: float = 0.5) -> None:
    """
    Compute and save confusion-matrix plots using a hard threshold on scores.

    For each class this is a binary problem (one-vs-rest):
        TP  predicted positive & actually positive
        FP  predicted positive & actually negative
        FN  predicted negative & actually positive
        TN  predicted negative & actually negative

    Saves:
        cfm_per_class.png  — one 2×2 heatmap per class in a grid
        cfm_all_classes.png — single C×C matrix: rows = true class,
                              cols = predicted class (argmax of scores),
                              restricted to clips that contain at least
                              one positive label, so background clips
                              don't flood the diagonal.
    """
    import math

    class_names = list(classes.keys())
    num_classes  = len(class_names)
    preds        = (scores >= threshold).astype(int)   # (N, C) binary

    cfm_dir = os.path.join(save_dir, 'confusion_matrices')
    os.makedirs(cfm_dir, exist_ok=True)

    # ── Per-class binary confusion matrices ───────────────────────────────────
    ncols   = 4
    nrows   = math.ceil(num_classes / ncols)
    fig1, axes1 = plt.subplots(nrows, ncols,
                                figsize=(ncols * 3, nrows * 3),
                                constrained_layout=True)
    fig1.suptitle(f'Per-class confusion matrices (threshold={threshold})',
                  fontsize=13)
    axes1_flat = axes1.flatten() if hasattr(axes1, 'flatten') else [axes1]

    for i, cls in enumerate(class_names):
        y_true = labels[:, i]
        y_pred = preds[:, i]

        TP = int(((y_pred == 1) & (y_true == 1)).sum())
        FP = int(((y_pred == 1) & (y_true == 0)).sum())
        FN = int(((y_pred == 0) & (y_true == 1)).sum())
        TN = int(((y_pred == 0) & (y_true == 0)).sum())

        mat   = np.array([[TP, FN], [FP, TN]], dtype=float)
        total = mat.sum()
        ax    = axes1_flat[i]

        im = ax.imshow(mat / (total + 1e-9), cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred +', 'Pred −'], fontsize=7)
        ax.set_yticklabels(['True +', 'True −'], fontsize=7)
        ax.set_title(cls, fontsize=8, pad=3)

        labels_mat = [['TP', 'FN'], ['FP', 'TN']]
        values_mat = [[TP, FN], [FP, TN]]
        for r in range(2):
            for c in range(2):
                ax.text(c, r,
                        f"{labels_mat[r][c]}\n{values_mat[r][c]}",
                        ha='center', va='center', fontsize=7,
                        color='white' if mat[r, c] / (total + 1e-9) > 0.5
                              else 'black')

    for j in range(num_classes, len(axes1_flat)):
        axes1_flat[j].set_visible(False)

    p1 = os.path.join(cfm_dir, 'cfm_per_class.png')
    fig1.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f'  Per-class confusion matrices saved → {p1}')

    # ── Multi-class confusion matrix (argmax prediction) ─────────────────────
    # Only consider clips that have at least one positive label
    has_label   = labels.sum(axis=1) > 0           # (N,) bool
    labels_sub  = labels[has_label]                 # (M, C)
    scores_sub  = scores[has_label]                 # (M, C)

    true_cls  = np.argmax(labels_sub,  axis=1)     # pick dominant true class
    pred_cls  = np.argmax(scores_sub,  axis=1)     # pick highest-scoring class

    cfm_all = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_cls, pred_cls):
        cfm_all[t, p] += 1

    # Normalise per row (recall-normalised)
    row_sums  = cfm_all.sum(axis=1, keepdims=True).astype(float)
    cfm_norm  = cfm_all / (row_sums + 1e-9)

    fig2, ax2 = plt.subplots(figsize=(num_classes * 0.9 + 1, num_classes * 0.9 + 1),
                              constrained_layout=True)
    im2 = ax2.imshow(cfm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax2.set_xticks(range(num_classes))
    ax2.set_yticks(range(num_classes))
    ax2.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
    ax2.set_yticklabels(class_names, fontsize=7)
    ax2.set_xlabel('Predicted class', fontsize=9)
    ax2.set_ylabel('True class', fontsize=9)
    ax2.set_title('Multi-class confusion matrix\n(row-normalised, argmax prediction)',
                  fontsize=11)

    for r in range(num_classes):
        for c in range(num_classes):
            ax2.text(c, r, f"{cfm_all[r, c]}",
                     ha='center', va='center', fontsize=6,
                     color='white' if cfm_norm[r, c] > 0.5 else 'black')

    p2 = os.path.join(cfm_dir, 'cfm_all_classes.png')
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'  Multi-class confusion matrix saved → {p2}')


def _serialisable(ap_history):
    """Convert numpy arrays inside ap_history dicts to plain lists for JSON."""
    out = []
    for entry in ap_history:
        clean = {}
        for k, v in entry.items():
            clean[k] = v.tolist() if isinstance(v, np.ndarray) else v
        out.append(clean)
    return out


if __name__ == '__main__':
    main(get_args())