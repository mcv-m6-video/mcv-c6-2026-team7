"""Small W&B wrapper for Week7 experiments."""

import wandb


def _to_serializable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]

    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}

    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass

    return str(value)


class WandbLogger:
    """Always-initialized W&B logger configured from JSON config plus runtime args."""

    def __init__(self, config, args):
        project = config.get('wandb_project', 'mcv-c6-week7-spotting')
        entity = config.get('wandb_entity')
        mode = config.get('wandb_mode', 'online')

        run_config = {
            'model_name': args.model,
            'save_metric': args.save_metric,
        }

        for key, value in config.items():
            run_config[f'config/{key}'] = _to_serializable(value)

        for key, value in vars(args).items():
            run_config[f'args/{key}'] = _to_serializable(value)

        init_kwargs = {
            'project': project,
            'mode': mode,
            'name': args.model,
            'config': run_config,
        }
        if entity is not None:
            init_kwargs['entity'] = entity

        self._run = wandb.init(**init_kwargs)

    def log_epoch(self, metrics):
        wandb.log(_to_serializable(metrics))

    def log_final(self, metrics):
        wandb.log(_to_serializable(metrics))

    def finish(self):
        if self._run is not None:
            wandb.finish()
            self._run = None
