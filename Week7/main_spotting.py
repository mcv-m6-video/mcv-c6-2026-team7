#!/usr/bin/env python3
"""
File containing the main training script.
"""

#Parse arguments
import argparse
import os
import sys
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--save_metric', type=str, default='map10_1', 
                        choices=['val_loss', 'map10_1', 'map10_0.5'], 
                        help='Metric used to save the best checkpoint')
    return parser.parse_args()

args = get_args()

# Set CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

#Standard imports
import importlib
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
from torch.utils.data import DataLoader
from tabulate import tabulate

#Local imports
from util.io import load_json, store_json
from util.eval_spotting import evaluate
from util.early_stopping import EarlyStopping
from util.wandb_logger import WandbLogger
from dataset.datasets import get_datasets

AP10_EXCLUDED = {'FREE KICK', 'GOAL'}
SUPPORTED_METRICS = {'val_loss', 'map10_1', 'map10_0.5'}
SUPPORTED_MODEL_MODULES = {'residual_bigru_TGLS', 'temporal_transformer_TGLS', 'unet', 'unet_ale', 'unet_transposed'}

def compute_mAP(ap_score, classes, exclude=None):
    excluded = set() if exclude is None else set(exclude)
    values = [
        ap_score[i]
        for i, class_name in enumerate(classes.keys())
        if class_name not in excluded
    ]
    return float(np.mean(values)) if values else 0.0

def update_args(args, config):
    #Update arguments with config file
    args.model_module = config.get('model_module', 'residual_bigru_TGLS')
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model # + '-' + str(args.seed) -> in case multiple seeds
    args.store_dir = config['save_dir'] + '/' + "splits"
    args.labels_dir = config['labels_dir']
    args.store_mode = config['store_mode']
    args.task = config['task']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.dataset = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.device = config['device']
    args.num_workers = config['num_workers']

    args.backbone_pretrained = config.get('backbone_pretrained', True)
    args.freeze_backbone = config.get('freeze_backbone', False)
    args.transformer_layers = config.get('transformer_layers', 3)
    args.transformer_dropout = config.get('transformer_dropout', 0.25)
    args.transformer_nhead = config.get('transformer_nhead', 8)
    args.label_smoothing_window = config.get('label_smoothing_window', 5)
    args.label_smoothing_sigma = config.get('label_smoothing_sigma', 0.55)
    args.save_metric = config.get('save_metric', args.save_metric)
    args.early_stopping_metric = config.get('early_stopping_metric', 'map10_1')
    args.early_stopping_patience = config.get('early_stopping_patience', 5)
    args.unet_dropout = config.get('unet_dropout', 0.0)
    args.label_smoothing = config.get('label_smoothing', 'none')
    args.label_smo_window = config.get('label_smo_window', 5)
    args.LS_gaussian_sigma = config.get('LS_gaussian_sigma', 0.55)

    return args

def validate_model_module(args):
    if args.model_module not in SUPPORTED_MODEL_MODULES:
        raise ValueError('Unsupported model_module "{}". Expected one of: {}'.format(args.model_module, sorted(SUPPORTED_MODEL_MODULES)))

def build_model(args):
    model_module = importlib.import_module('model.' + args.model_module)
    model_class = getattr(model_module, 'Model')
    return model_class(args=args)

def validate_metric(metric_name, field_name):
    if metric_name not in SUPPORTED_METRICS:
        raise ValueError('Unsupported {} "{}". Expected one of: {}'.format(field_name, metric_name, sorted(SUPPORTED_METRICS)))

def get_metric_value(metric_name, val_loss, val_map10_05, val_map10_10):
    metric_values = {
        'val_loss': val_loss,
        'map10_0.5': val_map10_05,
        'map10_1': val_map10_10,
    }
    return metric_values[metric_name]

def is_better(metric_name, current_value, best_value):
    if metric_name == 'val_loss':
        return current_value < best_value
    return current_value > best_value

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    num_epochs = int(args.num_epochs)
    warmup_epochs = min(int(args.warm_up_epochs), num_epochs)
    cosine_epochs = num_epochs - warmup_epochs

    if warmup_epochs > 0 and cosine_epochs > 0:
        print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
            warmup_epochs, cosine_epochs))
        return num_epochs, ChainedScheduler([
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                     total_iters=warmup_epochs * num_steps_per_epoch),
            CosineAnnealingLR(optimizer,
                num_steps_per_epoch * cosine_epochs)])

    if warmup_epochs > 0:
        print('Using Linear Warmup ({}) only'.format(warmup_epochs))
        return num_epochs, LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs * num_steps_per_epoch,
        )

    print('Using Cosine Annealing LR ({}) without warmup'.format(cosine_epochs))
    return num_epochs, CosineAnnealingLR(
        optimizer,
        num_steps_per_epoch * cosine_epochs,
    )

def plot_metrics(metrics_log, save_dir, model_name):
    epochs_list = [m['epoch'] for m in metrics_log]
    train_losses = [m['train_loss'] for m in metrics_log]
    val_losses = [m['val_loss'] for m in metrics_log]
    val_map05 = [m['val_map10_0.5'] for m in metrics_log]
    val_map10 = [m['val_map10_1.0'] for m in metrics_log]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Losses
    color_loss = '#d62728' 
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', color=color_loss, fontweight='bold')
    line1, = ax1.plot(epochs_list, train_losses, color=color_loss, linestyle='--', linewidth=2, label='Train Loss')
    line2, = ax1.plot(epochs_list, val_losses, color=color_loss, linestyle='-', linewidth=2, label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, alpha=0.3)

    # mAP
    ax2 = ax1.twinx()  
    color_map = '#1f77b4' 
    ax2.set_ylabel('mAP10', color=color_map, fontweight='bold')
    line3, = ax2.plot(epochs_list, val_map05, color=color_map, linestyle='--', linewidth=2, label='Val mAP10@0.5')
    line4, = ax2.plot(epochs_list, val_map10, color=color_map, linestyle='-', linewidth=2, label='Val mAP10@1.0')
    ax2.tick_params(axis='y', labelcolor=color_map)

    # Title and Legends
    plt.title(f'Training Metrics Evolution ({model_name})', fontweight='bold', pad=15)
    
    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', bbox_to_anchor=(1.35, 0.5))

    fig.tight_layout() 
    
    plt.savefig(os.path.join(save_dir, 'metrics_plot.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def main(args):
    # Print GPU info
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Check --gpu argument and drivers.")
    gpu_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_id)
    print(f"Using GPU {gpu_id}: {gpu_name}  (CUDA_VISIBLE_DEVICES='{args.gpu}')")

    # Set seed
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)
    validate_model_module(args)
    validate_metric(args.save_metric, 'save_metric')
    validate_metric(args.early_stopping_metric, 'early_stopping_metric')

    wandb_logger = WandbLogger(config=config, args=args)

    # Directory for storing / reading model checkpoints
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, train_data, val_data, val_video_data, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    # Model
    model = build_model(args)

    if not args.only_test:
        optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})
    
        # Dataloaders
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
        # Warmup schedule
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch)
        
        metrics_log = []
        best_criterion = float('inf') if args.save_metric == 'val_loss' else -float('inf')
        stop_mode = 'min' if args.early_stopping_metric == 'val_loss' else 'max'
        early_stopper = EarlyStopping(mode=stop_mode, patience=args.early_stopping_patience)
        epoch = 0

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):

            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler)
            
            val_loss = model.epoch(val_loader)
            val_mAP_dict, val_AP_per_class_dict = evaluate(model, val_video_data, nms_window=5)
            val_map10_05 = compute_mAP(val_AP_per_class_dict[0.5], classes, exclude=AP10_EXCLUDED)
            val_map10_10 = compute_mAP(val_AP_per_class_dict[1.0], classes, exclude=AP10_EXCLUDED)

            current_save_metric = get_metric_value(args.save_metric, val_loss, val_map10_05, val_map10_10)

            better = False
            if is_better(args.save_metric, current_save_metric, best_criterion):
                best_criterion = current_save_metric
                better = True

            current_stop_metric = get_metric_value(args.early_stopping_metric, val_loss, val_map10_05, val_map10_10)
            _, should_stop = early_stopper.update(current_stop_metric)
            
            #Printing info epoch
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f} | Val mAP10@0.5: {:0.5f} | Val mAP10@1.0: {:0.5f}'.format(
                epoch, train_loss, val_loss, val_map10_05, val_map10_10))
            if better:
                print(f'New best model epoch based on {args.save_metric}!')

            metrics_item = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_map12_0.5': float(val_mAP_dict[0.5]),
                'val_map12_1.0': float(val_mAP_dict[1.0]),
                'val_map10_0.5': val_map10_05,
                'val_map10_1.0': val_map10_10,
                'save_metric_value': current_save_metric,
                'early_stopping_metric_value': current_stop_metric
            }
            metrics_log.append(metrics_item)

            wandb_logger.log_epoch({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_map12_0.5': float(val_mAP_dict[0.5]),
                'val_map12_1.0': float(val_mAP_dict[1.0]),
                'val_map10_0.5': val_map10_05,
                'val_map10_1.0': val_map10_10,
                'save_metric_value': current_save_metric,
                'early_stopping_metric_value': current_stop_metric,
                'lr': optimizer.param_groups[0]['lr']
            })

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'metrics.json'), metrics_log, pretty=True)
                plot_metrics(metrics_log, args.save_dir, args.model)

                if better:
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt'))

            if should_stop:
                print('Early stopping triggered on {} with patience={}.'.format(args.early_stopping_metric, args.early_stopping_patience))
                break

    print('START INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))

    # Model size and MACs
    num_params = sum(p.numel() for p in model._model.parameters())
    try:
        from thop import profile
        dummy_input = torch.zeros(
            1, args.clip_len, 3, 224, 398, device=model.device)
        macs, _ = profile(model._model, inputs=(dummy_input,), verbose=False)
        macs_str = f"{macs/1e9:.2f} GMACs"
    except Exception as e:
        macs_str = "N/A (install thop: pip install thop)"
        print(f"Thop failed with error: {e}")

    print(f'\nModel params: {num_params:,}  |  MACs: {macs_str}')

    # Evaluation on test split
    mAP_dict, AP_per_class_dict = evaluate(model, test_data, nms_window=5)
    
    # Get the sorted list of tolerances
    tolerances = sorted(mAP_dict.keys())

    # Compute mAP10 for each tolerance
    map10_dict = {}
    for delta in tolerances:
        map10_dict[delta] = compute_mAP(AP_per_class_dict[delta], classes, exclude=AP10_EXCLUDED)

    # Report results per-class in table
    table = []
    for i, class_name in enumerate(classes.keys()):
        excluded = " (*)" if class_name in AP10_EXCLUDED else ""
        row = [class_name + excluded]
        for delta in tolerances:
            row.append(f"{AP_per_class_dict[delta][i]*100:.2f}")
        table.append(row)

    headers = ["Class"] + [f"AP @ {delta}s" for delta in tolerances]
    print(tabulate(table, headers, tablefmt="grid"))

    # Report mAP12 and mAP10 in table
    avg_table = []
    row_map12 = ["mAP12 (all classes)"]
    row_map10 = ["mAP10 (excl. FREE KICK & GOAL)"]
    
    for delta in tolerances:
        row_map12.append(f"{mAP_dict[delta]*100:.2f}")
        row_map10.append(f"{map10_dict[delta]*100:.2f}")
        
    avg_table.append(row_map12)
    avg_table.append(row_map10)

    avg_headers = ["Metric"] + [f"mAP @ {delta}s" for delta in tolerances]
    print(tabulate(avg_table, avg_headers, tablefmt="grid"))
    print("(*) excluded from mAP10")

    wandb_summary = {
        'num_params': num_params,
    }
    for delta in tolerances:
        wandb_summary[f'test_map12_{delta}'] = float(mAP_dict[delta])
        wandb_summary[f'test_map10_{delta}'] = float(map10_dict[delta])
    wandb_logger.log_final(wandb_summary)
    
    print('CORRECTLY FINISHED TRAINING AND INFERENCE')

    wandb_logger.finish()

if __name__ == '__main__':
    main(args)