import math

from catex.schedulers.scheduler import SchedulerFixedStop
from catex.schedulers.scheduler import SchedulerEarlyStop


class SchedulerTransformer(SchedulerFixedStop):
    """
    The standard scheduler of "Attention is all you need"
    """

    def __init__(
            self,
            optimizer_configs,
            paramlist,
            d_model: int,
            warmup_steps: int,
            epoch_max: int,
            reverse_metric_direc=False):
        super().__init__(optimizer_configs, paramlist, epoch_max, reverse_metric_direc)
        assert d_model > 0
        assert warmup_steps > 0
        self.lr_init = 1./math.sqrt(d_model)
        self._div_warmup_steps = 1./math.sqrt(warmup_steps)/warmup_steps
        self.update_lr(1)

    def update_lr(self, global_step: int):
        """Update the learning rate with global step"""
        step = float(global_step)
        lr = self.lr_init * min(1./math.sqrt(step),
                                step*self._div_warmup_steps)
        self._adjust_lr_(lr)

    def custom_update(self):
        """Do nothing"""
        return None


class SchedulerTransformerEarlyStop(SchedulerEarlyStop):
    """
    Linear warmup by step + decay by step + early stop by epoch
    peak lr = peak_factor / sqrt(d_model)
    """

    def __init__(
            self,
            optimizer_configs,
            paramlist,
            peak_factor: float,
            d_model: int,
            warmup_steps: int,
            lr_stop=1e-5,
            num_ahead=1,
            gamma=0.1,
            reverse_metric_direc=False):
        super().__init__(optimizer_configs, paramlist, 0,
                         lr_stop, num_ahead, gamma, reverse_metric_direc)
        assert d_model > 0
        assert warmup_steps > 0
        self.lr_init = peak_factor/math.sqrt(d_model)
        self._div_warmup_steps = 1./math.sqrt(warmup_steps)/warmup_steps
        self.step_cur = 0
        self.warmup_steps = warmup_steps
        self.update_lr(1)

    def update_lr(self, global_step: int):
        """Update the learning rate with global step"""
        self.step_cur = global_step
        step = float(global_step)
        lr = self.lr_init * min(1./math.sqrt(step),
                                step*self._div_warmup_steps)
        self._adjust_lr_(lr)

    def impl_step(self, metric):
        if self.step_cur <= self.warmup_steps:
            if not (self._reverse_ ^ (metric < self.best_metric)):
                self.best_metric = metric
            
            return 0
        else:
            lr0 = self.lr_cur
            states = super().impl_step(metric)
            lr1 = self.lr_cur
            self.lr_init *= lr1 / lr0
            return states