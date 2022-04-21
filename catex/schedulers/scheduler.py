import torch
import math
import numpy as np

from typing import Iterable
from typing import Union
from typing import Tuple

from collections import OrderedDict

if torch.__version__ >= '1.8.0':
    from torch.distributed.optim import ZeroRedundancyOptimizer

def SetupOptim(type_optim: str,
               paramlist: Iterable[torch.nn.parameter.Parameter],
               use_zero: bool = False,
               **kwargs) -> Union[torch.optim.Optimizer, ZeroRedundancyOptimizer]:
    """
    Setup the optimizer.

    Args:
        type_optim (str): name of optimizer, should be an attribute of `torch.optim`,
                like `Adam`, `SGD`
        paramlist (Iterable[Parameter]): a iterator or generator that returns parameters 
                to be optimized.
        use_zero (bool, default False): a flag to determinite whether use `ZeroRedundancyOptimizer`
                or not, ref to https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html,
                which is supported since torch 1.8.0
        
        **kwargs: any keyword arguments can be passed into optimizer initialization.
    Return:
        optimizer (torch.optim.Optimizer | ZeroRedundancyOptimizer)

    Example:
        >>> # with `use_zero=False`
        >>> model = nn.Linear(3,4)
        >>> optimizer = SetupOptim('Adam', model.parameters(), lr=1e-3, betas = (0.9, 0.99))
        >>> # with `use_zero=True`
        >>> #.... (init of DDP)
        >>> model = torch.nn.parallel.DistributedDataParallel(model)
        >>> optimizer = SetupOptim('Adam', model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    """
    if not use_zero:
            return getattr(torch.optim, type_optim)(paramlist, **kwargs)
    else:
        raise NotImplementedError


class Scheduler(object):
    def __init__(self,
                 optimizer_configs: dict,
                 paramlist: Iterable[torch.nn.parameter.Parameter],
                 reverse_metric_direc = False):
        super(Scheduler, self).__init__
        self.optimizer = SetupOptim(
            optimizer_configs['type_optim'],
            paramlist,
            **optimizer_configs['optim_conf']
        )

        self.epoch_cur = 0
        self.best_metric = None
        self._reverse_ = reverse_metric_direc
        self.lr_init = self.lr_cur

    @property
    def lr_cur(self):
        return self.optimizer.param_groups[0]['lr']
    
    def update_lr(self, *args, **kwargs):
        return None
    
    def _adjust_lr_(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        output = OrderedDict()
        for name, value in vars(self).items():
            if name == 'optimizer':
                output['optimizer'] = value.state_dict()
            else:
                output[name] = value
        return output

    def load_state_dict(self, 
                        ckpt: OrderedDict,
                        optim_only: bool = False):
        if optim_only:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            return None

        for name in vars(self).keys():
            if name not in ckpt:
                continue
            if name == "optimizer":
                self.optimizer.load_state_dict(ckpt[name])
            else:
                setattr(self, name, ckpt[name])

    def impl_step(self, metric) -> Tuple[int, str]:
        raise NotImplementedError
    
    def step(self,
             global_epoch: int,
             metric) -> Tuple[int, str]:
        """Optimizer step

        Args:
            global_epoch(int): the global epoch (begings from 1)
            metric (obj): the metric for evalute the performance


        Returns: (state, info)
            state(int): choice of `[0,1,2]`, meaning
                0: continue training by the prior condition
                1: contunue training for metirc is improving
                2: stop training.

            info(str): information
        """
        if self.best_metric is None:
            self.best_metric = metric
        
        self.epoch_cur = global_epoch
        return self.impl_step(metric)

class SchedulerEarlyStop(Scheduler):
    def __init__(
        self,
        optimizer_configs,
        paramlist,
        epoch_min: int,
        lr_stop: float = 1e-5,
        num_ahead: int = 1,
        gamma: float = 0.1,
        reverse_metric_direc = False
    ):
        super().__init__(
                        optimizer_configs,
                        paramlist,
                        reverse_metric_direc)

        self.lr_stop = lr_stop
        self.epoch_min = epoch_min
        self.num_ahead = num_ahead
        self.gamma = gamma
        self.count_worse = 0

    def impl_step(self, metric):

        state = 0
        info = ''
        if self.epoch_cur <= self.epoch_min:
            if not (self._reverse_ ^ (metric < self.best_metric)):
                self.best_metric = metric
                state = 1
        elif not (self._reverse_ ^ (metric < self.best_metric)):
            self.best_metric = metric
            self.count_worse = 0
            state = 1
        else:
            self.count_worse += 1
            if self.count_worse >= self.num_ahead:
                lr = self.lr_cur
                lr *= self.gamma
                if lr < self.lr_stop:
                    state = 2
                else:
                    self._adjust_lr_(lr)
                    self.count_worse = 0

        return state


class SchedulerFixedStop(Scheduler):
    def __init__(
            self,
            optimizer_configs,
            paramlist,
            epoch_max: int,
            reverse_metric_direc=False):
        super().__init__(optimizer_configs, paramlist, reverse_metric_direc)
        self.epoch_max = epoch_max

    def custom_update(self):
        return None

    def impl_step(self, metric):
        state = 0
        if not (self._reverse_ ^ (metric < self.best_metric)):
            self.best_metric = metric
            state = 1
        elif self.epoch_cur >= self.epoch_max:
            state = 2

        self.custom_update()

        return state
