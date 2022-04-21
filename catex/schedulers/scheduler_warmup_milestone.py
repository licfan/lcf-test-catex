from catex.schedulers.scheduler import SchedulerEarlyStop

class SchedulerWarmupMileStone(SchedulerEarlyStop):
    """MileStone scheduler with warmup
        
    Combine the linear warmup and mile stone decreasing up
    """

    def __init__(
            self,
            optimizer_configs,
            paramlist,
            total_batch_size: int,
            warmup_epoch: int,
            refer_batch: int,
            refer_lr: float = 0.,
            lr_stop=1e-5,
            num_ahead=1,
            gamma=0.1,
            reverse_metric_direc=False):
        super().__init__(optimizer_configs, paramlist, 0, lr_stop,
                         num_ahead, gamma, reverse_metric_direc)
        if refer_lr == 0.:
            refer_lr = self.lr_init

        assert total_batch_size > 0
        assert warmup_epoch > 0
        assert refer_batch > 0
        assert refer_lr > 0

        self.max_lr = max(total_batch_size/refer_batch * refer_lr, refer_lr)
        if self.lr_init != refer_lr:
            print("Warning: the learning set in optimizer and `refer_lr` are different.")
            self.lr_init = refer_lr
            self._adjust_lr_(refer_lr)

        self.epoch_warmup = warmup_epoch
        self.lr_addon = (self.max_lr-self.lr_init)/warmup_epoch

    def impl_step(self, metric):
        if self.epoch_cur <= self.epoch_warmup:
            state = 0
            if not (self._reverse_ ^ (metric < self.best_metric)):
                self.best_metric = metric
                state = 1
            cur_lr = self.lr_cur
            self._adjust_lr_(cur_lr+self.lr_addon)
            info = "Epoch: [{}/{}] | best={:.2f} | current={:.2f} | lr={:.2e}".format(
                self.epoch_cur, self.epoch_warmup, self.best_metric, metric, self.lr_cur)
            return state, info
        else:
            return super().impl_step(metric)