import numpy as np

from catex.schedulers.scheduler import SchedulerFixedStop


class SchedulerIterAnnealing(SchedulerFixedStop):
    def __init__(
            self,
            optimizer_configs,
            paramlist,
            decay_factor: float,
            epoch_max: int,
            reverse_metric_direc=False):
        super().__init__(optimizer_configs, paramlist, epoch_max, reverse_metric_direc)
        assert decay_factor > 0. and decay_factor < 1. and epoch_max > 0
        self.decay = decay_factor

    def custom_update(self):
        lr = self.lr_init * (self.decay ** self.epoch_cur)
        self._adjust_lr_(lr)


class SchedulerCosineAnnealing(SchedulerFixedStop):
    def __init__(
            self,
            optimizer_configs,
            paramlist,
            lr_min: float,
            epoch_max: int,
            period: int = 0,
            decay_factor: float = 1.,
            reverse_metric_direc=False):
        super().__init__(optimizer_configs, paramlist, epoch_max, reverse_metric_direc)
        assert period >= 0 and lr_min >= 0 and epoch_max > 0
        assert decay_factor > 0. and decay_factor <= 1.
        if period == 0:
            period = epoch_max

        self.period = period
        self.decay = decay_factor
        self.lr_min = lr_min
        self.lr_max = self.lr_init

    def custom_update(self):
        epoch_idx = self.epoch_cur - 1
        lr_max = (self.lr_max *
                  self.decay**(epoch_idx//self.period))

        lr = self.lr_min + 0.5 * (lr_max - self.lr_min) * (
            1 + np.cos((epoch_idx % self.period)/self.period * np.pi))
        self._adjust_lr_(lr)
