import torch

from typing import Union
from typing import Optional
from pathlib import Path

from contextlib import contextmanager
from distutils.version import LooseVersion

import argparse
import time
import math
import logging
import humanfriendly
import torch.distributed as dist

from catex.torch_utils.set_random_seed import set_random_seed

import dataclasses

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from catex.train.reporter import Reporter
from catex.train.reporter import SubReporter
from catex.schedulers.scheduler import Scheduler
from catex.utils.build_dataclass import build_dataclass

from typeguard import check_argument_types

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
    from torch.cuda.amp import GradScaler
else:
    @contextmanager
    def autocast(enabled=True):
        yield

    GradScaler = None

@dataclasses.dataclass
class AsrTrainerOptions:
    resume: bool
    grad_accum_fold: int
    output_dir: Union[str, Path]
    use_tensorboard: bool
    use_matplotlib: bool
    max_epoch: int
    seed: int
    log_interval: int
    valid_interval: int
    amp: bool

class AsrTrainer:
    def __init__(self):
        raise RuntimeError("This class can't be instantiated")

    @classmethod
    def build_options(cls, args: argparse.Namespace) -> AsrTrainerOptions:
        assert check_argument_types()
        return build_dataclass(AsrTrainerOptions, args)
 
    @classmethod
    def resume(
        cls,
        checkpoint: Union[str, Path],
        model: torch.nn.Module,
        reporter: Reporter,
        scheduler: Scheduler,
        scaler: Optional[GradScaler],
        ngpu: int = 0,
    ):
        states = torch.load(
            checkpoint,
            map_location = f"cuda:{ngpu}"
        )
        dist.barrier()
        model.load_state_dict(states["model"])
        scheduler.load_state_dict(states["scheduler"])
        reporter.load_state_dict(states["reporter"])
        if scaler is not None:
            if states["scaler"] is None:
                logging.warning("scaler state is not found")
            else:
                scaler.load_state_dict(states["scaler"])
        logging.info(f"The training is resumed using {checkpoint}")


    @classmethod
    def run(
        cls,
        trainer_options: AsrTrainerOptions,
        train_sampler: DistributedSampler,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model: torch.nn.parallel.distributed.DistributedDataParallel,
        scheduler: Scheduler,
        is_crf: bool,
        process_id: int,
        ):

        assert check_argument_types()
        output_dir = Path(trainer_options.output_dir)

        if trainer_options.amp:
            if GradScaler == None:
                raise RuntimeError(f"amp requires to be larger than 1.6, and yours is {torch.__version__}")
            scaler = GradScaler()
        else:
            scaler = None
    
        reporter = Reporter(batch_num = len(train_loader))
        if trainer_options.resume and (output_dir / "checkpoint.pth").exists():
            cls.resume(
                checkpoint=output_dir / "checkpoint.pth",
                model = model,
                reporter = reporter,
                scheduler = scheduler,
                scaler = scaler,
                ngpu=process_id,
            )

        start_epoch = reporter.get_epoch() + 1

        if start_epoch == trainer_options.max_epoch + 1:
            logging.warning(
                f"The training has already reached at max_epoch: {start_epoch}"
            )
        
        fold = trainer_options.grad_accum_fold
        assert fold >= 1

        if trainer_options.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(str(output_dir / "tensorboard"))
        else:
            summary_writer = None
        
        start_time = time.perf_counter()
        for iepoch in range(start_epoch, trainer_options.max_epoch + 1):
            if process_id == 0:
                if iepoch != start_epoch:
                    logging.info(
                        "{}/{} epoch started. Estimated time to finish: {}".format(
                            iepoch,
                            trainer_options.max_epoch,
                            humanfriendly.format_timespan(
                                (time.perf_counter() - start_time)
                                / (iepoch - start_epoch)
                                * (trainer_options.max_epoch - iepoch + 1)
                            ),
                        )
                    )
                else:
                    logging.info(f"{iepoch}/{trainer_options.max_epoch} epoch started")
        
            set_random_seed(trainer_options.seed + iepoch)
            reporter.set_epoch(iepoch)
            train_sampler.set_epoch(iepoch)
            with reporter.observe("train") as sub_reporter:
                cls.train_one_epoch(
                    train_loader = train_loader,
                    model = model,
                    scheduler = scheduler,
                    trainer_options = trainer_options,
                    process_id = process_id,
                    is_crf = is_crf,
                    reporter = sub_reporter,
                    scaler = scaler,
                    summary_writer=summary_writer
                )
            with reporter.observe("test") as sub_reporter:
                real_loss_avg = cls.validate_one_epoch(
                    model = model,
                    test_loader = test_loader,
                    process_id = process_id,
                    is_crf = is_crf,
                    reporter = sub_reporter
                )
            
             # Scheduler step
            state = scheduler.step(iepoch, real_loss_avg)

             # report results
            if process_id == 0:
                logging.info(reporter.log_message())
                if trainer_options.use_matplotlib:
                    reporter.matplotlib_plot(output_dir / "images")
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer)

                if state == 0 or state == 1:
                    # save/update checkpoint
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "reporter": reporter.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict() if scaler is not None else None
                        },
                        output_dir / "checkpoint.pth",
                    )
                    torch.save(model.state_dict(), output_dir / f"{iepoch}epoch.pth")

                    # Creates a sym link latest.pth -> {iepoch}epoch.pth
                    p = output_dir / "latest.pth"
                    if p.is_symlink() or p.exists():
                        p.unlink()
                    p.symlink_to(f"{iepoch}epoch.pth")

                    if state == 1:
                        best = output_dir / "best.pth"
                        if best.is_symlink() or best.exists():
                            best.unlink()
                        best.symlink_to(f"{iepoch}epoch.pth")

            if state == 2:
                dist.barrier()
                break

            torch.cuda.empty_cache()
        else:
            logging.info(
                f"The training was finished at {trainer_options.max_epoch} epochs"
            )

    @classmethod
    def train_one_epoch(
        cls, 
        train_loader: DataLoader,
        model: torch.nn.parallel.distributed.DistributedDataParallel,
        scheduler: Scheduler,
        trainer_options: AsrTrainerOptions,
        reporter: SubReporter,
        process_id: int,
        is_crf: bool,
        scaler: Optional[GradScaler],
        summary_writer,
    ):
        assert check_argument_types()

        @torch.no_grad()
        def _cal_real_loss(loss, path_weights, is_crf):
            if is_crf:
                partial_loss = loss.cpu()
                weight = torch.mean(path_weights)
                return partial_loss - weight
            else:
                return loss.cpu()

        log_interval = trainer_options.log_interval
        fold = trainer_options.grad_accum_fold
        optimizer = scheduler.optimizer

        model.train()

        start_time = time.perf_counter()
        epoch = reporter.get_epoch()

        pre_steps = int(math.ceil(len(train_loader)/float(fold)) * (epoch - 1))
 
        optimizer.zero_grad()
        for i, minibatch in enumerate(train_loader):
            lr_cur = scheduler.lr_cur
            reporter.register(dict(lr=lr_cur), scalar = True)
            # measure data loading time
            with reporter.measure_time("loading_time"):
                logits, input_lengths, labels, label_lengths, path_weights = minibatch
                logits, labels, input_lengths, label_lengths \
                = logits.cuda(process_id, non_blocking=True), labels, input_lengths, label_lengths

            if fold == 1 or (i + 1) % fold == 0 or (i + 1) == len(train_loader):
                # update every fold times and won't drop the last batch
                with autocast(scaler is not None):
                    with reporter.measure_time("forward_time"):
                        loss = model(logits, labels, input_lengths, label_lengths)
                with reporter.measure_time("backward_time"):
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                real_loss = _cal_real_loss(loss, path_weights, is_crf)

                # for Adam optimizer, even though fold > 1, it's no need to normalize grad
                # if using SGD, let grad = grad_accum / fold as following or use a new_lr = init_lr / fold
                # if fold > 1:
                #    for param in model.parameters():
                #       if param.requires_grad:
                #           param.grad.data /= fold

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.update_lr(pre_steps + (i + 1) / fold)
     
                # regist loss
                reporter.register(dict(losses=loss.item()), weight=logits.size(0))
                # regist real loss
                reporter.register(dict(losses_real=real_loss.item()),logits.size(0))
                # registe elapsed time
                reporter.register(
                    dict(
                            batch_time=time.perf_counter() - start_time
                    ),
                )
                start_time = time.perf_counter()

            else:
                # gradient accumulation w/o sync
                with model.no_sync():
                    with autocast(scaler is not None):
                        with reporter.measure_time("forward_time"):
                            loss = model(logits, labels, input_lengths, label_lengths)
                    with reporter.measure_time("backward_time"):
                        if scaler is not None:
                            scaler.scale(loss).backward
                        else:
                            loss.backward()
                        
            reporter.next()
            if (i + 1) %  log_interval == 0 and process_id == 0:
                logging.info(reporter.log_message(-log_interval))
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer, -log_interval)
            
    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls, 
        model: torch.nn.parallel.distributed.DistributedDataParallel,
        test_loader: DataLoader,
        process_id: int,
        is_crf: bool,
        reporter: SubReporter
    ):

        assert check_argument_types()
        model.eval()
        start_time = time.perf_counter()
        total_count = 0.0
        sum_real_loss = 0.0
        for i, minibatch in enumerate(test_loader):
            # measure data loading time
            logits, input_lengths, labels, label_lengths, path_weights = minibatch
            logits, labels, input_lengths, label_lengths = logits.cuda(
                process_id, non_blocking = True), labels, input_lengths, label_lengths

            path_weights = path_weights.cuda(process_id, non_blocking = True)

            with reporter.measure_time("inference_time"):
                loss = model(logits, labels, input_lengths, label_lengths)

            if is_crf:
                weight = torch.mean(path_weights)
                real_loss = loss - weight
            else:
                real_loss = loss

            dist.all_reduce(real_loss, dist.ReduceOp.SUM)
            real_loss = real_loss / dist.get_world_size()

            sum_real_loss += real_loss * logits.size(0)
            total_count += logits.size(0)
            # measure accuracy and record loss
            # regist loss
            reporter.register(dict(losses_real=real_loss.item()), weight=logits.size(0))
            # registe elapsed time
            reporter.register(
                dict(
                        batch_time=time.perf_counter() - start_time
                ),
            )
            start_time = time.perf_counter() - start_time
            reporter.next()

        real_loss_avg = sum_real_loss / total_count
        return real_loss_avg.item()
            