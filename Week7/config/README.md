# Setting up the Configurations

Here we describe the different parameters set in the baseline configuration file:

Week7 supports multiple model modules:
- `residual_bigru_TGLS` (Residual BiGRU temporal neck)
- `temporal_transformer_TGLS` (Transformer temporal neck)

Backbone changes should be done by changing `feature_arch` in config.

- _frame_dir:_ Directory where frames are stored.
- _save_dir:_ Directory to save checkpoints, dataset information, etc.
- _labels_dir:_ Directory where dataset labels are stored.
- _store_mode:_ 'store' if it's the first time running the script to prepare and store dataset information, or 'load' to load previously stored information.
- _task_: either 'classification' or 'spotting'
- _model_module:_ Model file to load (`residual_bigru_TGLS` or `temporal_transformer_TGLS`).
- _batch_size:_ Batch size.
- _clip_len:_ Length of the clips in number of frames.
- _stride:_ Sampling one out of every _stride_ frames when reading from _frame_dir_.
- _dataset:_ Name of the dataset ('soccernetball').
- _epoch_num_frames:_ Number of frames used per epoch.
- _feature_arch:_ Feature extractor architecture from the Week7 backbone builder in `model/modules.py` (currently `rny002`, `rny004`, `rny008`, `convnextv2_pico`, `convnextv2_atto`).
- _backbone_pretrained:_ Whether to load pretrained timm weights for the selected backbone.
- _freeze_backbone:_ If true, freezes backbone parameters.
- _transformer_layers:_ Number of Transformer encoder layers (used by `temporal_transformer_TGLS`).
- _transformer_dropout:_ Dropout used by Transformer encoder layers (used by `temporal_transformer_TGLS`).
- _transformer_nhead:_ Number of attention heads in the Transformer encoder (used by `temporal_transformer_TGLS`).
- _learning_rate:_ Learning rate.
- _num_classes:_ Number of classes for the current dataset.
- _num_epochs:_ Number of epochs for training.
- _warm_up_epochs:_ Number of warm-up epochs.
- _label_smoothing_window:_ Temporal Gaussian smoothing window for target labels.
- _label_smoothing_sigma:_ Temporal Gaussian smoothing sigma.
- _save_metric:_ Metric to select the best checkpoint (`val_loss`, `map10_1`, `map10_0.5`).
- _early_stopping_metric:_ Metric used by early stopping (`val_loss`, `map10_1`, `map10_0.5`).
- _early_stopping_patience:_ Number of non-improving epochs allowed before stop.
- _wandb_project:_ Weights & Biases project name.
- _wandb_entity:_ Weights & Biases entity (team/user), or null.
- _wandb_mode:_ Weights & Biases mode (`online`, `offline`, `disabled`).
- _only_test:_ Boolean indicating whether only inference or training + inference.
- _device:_ Either "cuda" or "cpu".
- _num_workers:_ Number of workers.

You are free to create new configurations and add the necessary parameters once you modify the baseline. At the very least, you'll need to modify `frame_dir`, `save_dir`, and `labels_dir` as they are set to work in our own computation servers.
