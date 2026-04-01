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

#Local imports
import importlib
from util.io import load_json, store_json
from util.eval_classification import evaluate
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
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed',  type=int, default=1)
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
    args.num_workers     = config['num_workers']
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


def run_ap_eval(model, val_data, class_frequencies, prev_ap, classes, alpha=ALPHA):
    """
    Run evaluate(), compute the weighted-change metric, and print a summary.
    Returns (current_ap, progress_dict).  prev_ap may be None on the first call.
    """
    current_ap = evaluate(model, val_data)          # np.ndarray (num_classes,)

    if prev_ap is None:
        # First eval — no change to compute yet
        progress = {
            'weighted_change' : None,
            'per_class_change': np.zeros_like(current_ap),
            'weights'         : np.ones_like(current_ap) / len(current_ap),
            'mean_ap'         : float(np.mean(current_ap)),
            'delta_mean_ap'   : None,
        }
    else:
        progress = compute_weighted_ap_change(
            current_ap, prev_ap, class_frequencies, alpha=alpha
        )

    # ── pretty print ─────────────────────────────────────────────────────────
    table = []
    for i, cls in enumerate(classes.keys()):
        delta = progress['per_class_change'][i]
        w     = progress['weights'][i]
        table.append([
            cls,
            f"{current_ap[i]*100:.2f}",
            f"{delta*100:+.2f}" if prev_ap is not None else "—",
            f"{w:.4f}",
        ])
    print(tabulate(
        table,
        headers=["Class", "AP (%)", "ΔAP (%)", "Weight"],
        tablefmt="grid"
    ))

    wc = progress['weighted_change']
    print(f"  mAP: {progress['mean_ap']*100:.2f}%  |  "
          f"Δ mAP: {progress['delta_mean_ap']*100:+.2f}%  |  "
          f"Weighted Δ: {wc:+.5f}"
          if wc is not None else
          f"  mAP: {progress['mean_ap']*100:.2f}%  (first eval, no delta yet)")

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
                    model, val_data, class_frequencies, prev_ap, classes
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

    ap_score = evaluate(model, test_data)

    table = [[cls, f"{ap_score[i]*100:.2f}"]
             for i, cls in enumerate(classes.keys())]
    print(tabulate(table, headers=["Class", "AP (%)"], tablefmt="grid"))
    print(tabulate([["Average", f"{np.mean(ap_score)*100:.2f}"]],
                   headers=["", "AP (%)"], tablefmt="grid"))

    print('CORRECTLY FINISHED TRAINING AND INFERENCE')


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