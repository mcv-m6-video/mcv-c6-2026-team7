"""Early-stopping utility used by Week7 training loops."""


class EarlyStopping:
    """Track a metric and stop when it does not improve for N updates."""

    def __init__(
        self,
        mode='min',
        patience=5,
        min_delta=0.0,
        warmup_epochs=0,
    ):
        if mode not in {'min', 'max'}:
            raise ValueError('mode must be "min" or "max"')

        self.mode = mode
        self.patience = max(0, int(patience))
        self.min_delta = float(min_delta)
        self.warmup_epochs = max(0, int(warmup_epochs))

        self.best_value = None
        self.bad_epochs = 0
        self.update_count = 0

    def _is_improved(self, value):
        if self.best_value is None:
            return True

        if self.mode == 'min':
            return (self.best_value - value) > self.min_delta
        return (value - self.best_value) > self.min_delta

    def update(self, value):
        """
        Update state with the current metric value.

        Returns
        -------
        improved : bool
            True when the current value is considered better than the best.
        should_stop : bool
            True when early stopping condition is reached.
        """
        self.update_count += 1
        improved = self._is_improved(value)

        if improved:
            self.best_value = value
            self.bad_epochs = 0
            return True, False

        if self.update_count > self.warmup_epochs and self.patience > 0:
            self.bad_epochs += 1

        should_stop = self.patience > 0 and self.bad_epochs >= self.patience
        return False, should_stop
