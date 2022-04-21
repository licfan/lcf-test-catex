#!/usr/bin/env python3
import argparse
import sys

from typeguard import check_argument_types
from typeguard import check_return_type
from typing import Any
from typing import Dict

import torch
import torch.distributed as dist

from pathlib import Path

from catex.torch_utils.set_random_seed import set_random_seed
from catex.torch_utils.model_summary import model_summary

import logging
import os
import yaml

from catex.asr.catex_dataset import SpeechDataset
from catex.asr.catex_dataset import SpeechDatasetPickle
from catex.asr.catex_dataset import sorted_pad_collate

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from catex.train.class_choices import ClassChoices
from catex.asr.catex_model import CatexModel

from catex.nets.conformer import ConformerNet
from catex.nets.rnn import BLSTM
from catex.nets.rnn import LSTM
from catex.nets.rnn import VGGBLSTM

from catex.asr.specaug import SpecAug

from ctc_crf import CRFContext
from ctc_crf import CTC_CRF_LOSS
from ctc_crf import WARP_CTC_LOSS

from catex.train.asr_trainer import AsrTrainer

from catex.schedulers.scheduler import Scheduler
from catex.schedulers.scheduler_warmup_milestone import SchedulerWarmupMileStone
from catex.schedulers.scheduler_transformer import SchedulerTransformerEarlyStop
from catex.schedulers.scheduler_annealing import SchedulerIterAnnealing
from catex.schedulers.scheduler_annealing import SchedulerCosineAnnealing

import torch.multiprocessing as mp
reserve_list = ["net", "net_conf", "scheduler", "scheduler_conf"]

net_choices = ClassChoices(
    name="net",
    classes = dict(
        conformer = ConformerNet,
        lstm = LSTM,
        blstm = BLSTM,
        vgglstm = VGGBLSTM
    ),
    type_check = torch.nn.Module,
    default='conformer',
)


scheduler_choices = ClassChoices(
    name="scheduler",
    classes = dict(
        scheduler_transformer_early_stop=SchedulerTransformerEarlyStop,
        scheduler_warmup_milestone = SchedulerWarmupMileStone,
        scheduler_iter_annealing = SchedulerIterAnnealing,
        scheduler_consine_annealing = SchedulerCosineAnnealing,
    ),
    type_check = Scheduler,
    default = 'scheduler_transformer_early_stop'
)
class AsrTraining():

    class_choices_list = [
        net_choices,
        scheduler_choices,
    ]

    @classmethod
    def get_config(cls, args: argparse.Namespace, net_config: Dict[str, Any]) -> Dict[str, Any]:
        """Return configuration.
        
        This method is called by dump_config()
        """
        assert check_argument_types()
        config = vars(args)
        for key in reserve_list:
            config.pop(key)
        config["train_config"] = net_config
                
        return config

    @classmethod
    def dump_config(cls, args: argparse.Namespace, net_config: Dict[str, Any]) -> None:
        assert check_argument_types()
        config = cls.get_config(args, net_config)
        with (Path(args.output_dir) / "asr_conf.yaml").open("w", encoding="utf-8") as f:
            f.write(yaml.dump(config, allow_unicode=True))

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Asr training base parser")

        group = parser.add_argument_group(description="training related")
        group.add_argument(
            "--train_config",
            type=str,
            required=True,
            help="config file for trainging ,include  epoch , layer dim .etc"
        )

        group.add_argument(
            "--output_dir",
            type=str,
            default=None,
            metavar='Path',
            help="directory to save the log and model files."
        )

        group.add_argument(
            "--batch_size",
            type=int,
            default=256,
            help="mini-batch size (default: 256), this is the total batch "
            "size of all GPUS on the current node when using Distributed Data Parallel"
        )
        group.add_argument(
            "--seed",
            type=int,
            default=0,
            help="manual seed"
            )

        group.add_argument(
            "--resume",
            action="store_true",
            help="whether to start from checkpoint"
        )

        group.add_argument(
            "--grad_accum_fold",
            type=int,
            default=1,
            help="Utilize gradient accumulation for K times. Default: K = 1"
        )

        group.add_argument(
            "--amp",
            action="store_true",
            help="whther to use automatic mixed precision"
        )

        group.add_argument(
            "--use_tensorboard",
            type=bool,
            default=True,
            help = "Enable tensorboard logging"
        )

        group.add_argument(
            "--use_matplotlib",
            type=bool,
            default=True,
            help = "Enable matplotlib logging"
        )

        group.add_argument(
            "--max_epoch", 
            type=int,
            default=50,
            help = "The maximum number epoch to train"
        )

        group.add_argument(
            "--log_interval",
            type=int,
            default=100,
            help="Output logs every the number iterators in each epochs just in training phase"
        )

        group.add_argument(
            "--valid_interval",
            type=int,
            default=10,
            help="Output logs every the number iterators in valid phase"
        )

        group.add_argument(
            "--data_set",
            type=str,
            default=None,
            help="Location of training/testing data."
        )

        group.add_argument(
            "--train_set",
            type=str,
            default=None,
            help="Location of training data. Default: \
             <data_set>/[pickle|hdf5]/tr.[pickle|hdf5]"
        )
        
        group.add_argument(
            "--dev_set",
            type=str,
            default=None,
            help="Location of training data. Default: \
             <data_set>/[pickle|hdf5]/cv.[pickle|hdf5]"
        )

        group = parser.add_argument_group(description="Distributed training related")
       
        group.add_argument(
            "--distributed",
            type=bool,
            default=False,
            help="traing with distributed mode or not"
        )

        group.add_argument(
            "--rank",
            type=int,
            default=0,
            help="node rank for distributed training"
        )

        group.add_argument(
            "--dist-url",
            type=str,
            default = "tcp://127.0.0.1:12947",
            help = "url used to set up distributed training"
        )

        group.add_argument(
            "--dist-backend",
            type=str,
            default="nccl",
            help="distributed backend"
        )

        group.add_argument(
            "--world-size", 
            type=int,
            default=1,
            help="number of nodes for distributed traning"
        )

        parser.add_argument(
            "--h5py",
            action="store_true",
            help="Load data with H5py, defaultly use pickle(recommended)."
        )

        parser.add_argument(
            "--log_level",
            type= lambda x: x.upper(),
            default="INFO",
            choices=["ERROR", "WARNING", "INFO", "DEBUG", "NOTEST"],
            help="The verbose level of logging"
        )

        parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help="number of data loading workers (default: 1)"
        )

        for class_choices in cls.class_choices_list:
            class_choices.add_arguments(parser)
        
        assert check_return_type(parser)
        return parser

    @classmethod
    def build_model(cls,
                    configs,
                    train: bool = True,
                    ) -> torch.nn.Module:

        # Data augmentation for spectorgram
        specaug = SpecAug(configs['specaug_conf'])

        net_conf = configs['net']
        net = net_choices.get_class(net_conf['type'])
        am_model = net(**net_conf['net_conf'])

        if not train:
            return am_model

        if 'lossfn' not in net_conf:
            lossfn = 'crf'
            logging.info("warning: not specified lssfn in configuration,\
                            Default set to crf ")
        else:
            lossfn = net_conf['lossfn']

        if lossfn == 'crf':
            if 'lamb' not in net_conf:
                lamb = 0.01
                logging.info("Warning: not specified lamb in configuration, \
                    Defaultly set to 0.01")
            else:
                lamb = net_conf['lamb']

            loss_fn = CTC_CRF_LOSS(lamb=lamb)
        elif lossfn == "ctc":
            loss_fn = WARP_CTC_LOSS()
        else:
            raise ValueError(f"Unknow loss function: {lossfn}")
        
        catex_model = CatexModel(am_model, loss_fn, specaug)
        torch.cuda.set_device(cls.process_id)
        catex_model.cuda(cls.process_id)
        catex_model = torch.nn.parallel.DistributedDataParallel(
            catex_model, device_ids = [cls.process_id]
        )
        return catex_model

    @classmethod
    def load_pretrained_model(
        path: str,
        model: torch.nn.Module,
        scheduler: Scheduler,
        map_location: str = "cpu"
        ):

        dist.barrier()
        checkpoint = torch.load(path, map_location = map_location)
        model.load_state_dict(checkpoint['model'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    @classmethod
    def main_worker(
        cls,
        process_id: int,
        ngpus_per_node: int,
        args: argparse.Namespace
    ):
        assert check_argument_types()
        cls.process_id = process_id
        torch.cuda.set_device(process_id)

        rank = args.rank * ngpus_per_node + process_id

        logging.basicConfig(
            level=args.log_level,
            format=f"[{os.uname()[1].split('.')[0]}-{rank}]"
                f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",)
        

        logging.info(f"Use GPU: local [{cls.process_id}] | global [{rank}]")

        dist.init_process_group(
            backend = args.dist_backend,
            init_method = args.dist_url,
            world_size = args.world_size,
            rank=rank
        )

        real_batch_size = args.batch_size // ngpus_per_node

        if cls.process_id == 0:
            logging.info("Start data prepare")

        if args.h5py:
            data_format="hdf5"
            logging.debug("H5py reading might case error with Multi-GPUS")

            Dataset = SpeechDataset
            assert (args.trset is not None and args.devset is not None),\
             "With '--hdf5' option, you must specify data location \
              with '--trset' and '--devset'"
        else:
            data_format = "pickle"
            Dataset = SpeechDatasetPickle

        if args.train_set is None:
            train_set = os.path.join(args.data_set,
             f'{data_format}/tr.{data_format}')
        else:
            train_set = args.train_set

        if args.dev_set is None:
            cv_set = os.path.join(args.data_set,
             f'{data_format}/cv.{data_format}')
        else:
            cv_set = args.dev_set
        
        tr_set = Dataset(train_set)
        test_set = Dataset(cv_set)

        train_sampler = DistributedSampler(tr_set)
        test_sampler = DistributedSampler(test_set)
        test_sampler.set_epoch(1)

        train_loader = DataLoader(
                                tr_set, 
                                batch_size = real_batch_size,
                                shuffle = (train_sampler is None),
                                num_workers = args.workers,
                                pin_memory = True,
                                sampler = train_sampler,
                                collate_fn = sorted_pad_collate)

        test_loader = DataLoader(
                                test_set,
                                batch_size = real_batch_size,
                                shuffle = (test_sampler is None),
                                num_workers = args.workers,
                                pin_memory = True,
                                sampler = test_sampler,
                                collate_fn = sorted_pad_collate
                                )

        with open(args.train_config, 'r') as f:
            import re
            safe_loader = yaml.SafeLoader
            safe_loader.add_implicit_resolver(
                u'tag:yaml.org,2002:float',
                re.compile(u'''^(?:
                [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$''', re.X),
                list(u'-+0123456789.'))
            configs = yaml.load(f, Loader = safe_loader)


        #dump config to outputdir/asr_conf.yaml
        cls.dump_config(args, configs)

        set_random_seed(args.seed)

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # build model
        model = cls.build_model(configs, True)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with (output_dir / "am_config").open("w", encoding="utf-8") as f:
            logging.info(
                f'Saving model configuration in {output_dir / "am_config"}'
            )
            f.write(model_summary(model))
        print(model)

        if configs['net']['lossfn'] == 'crf':
            ctx = CRFContext(f"{args.data_set}/den_meta/den_lm.fst", cls.process_id)

        # initial scheduler and optimizer
        scheduler_class = scheduler_choices.get_class(configs['scheduler']['type'])
        scheduler_conf = configs['scheduler']['scheduler_conf']

        optimizer_conf = configs['scheduler']['optimizer']

        scheduler = scheduler_class(optimizer_conf,
             model.parameters(), **scheduler_conf)
    
        # start training
        asr_trainer_option = AsrTrainer.build_options(args)

        AsrTrainer.run(asr_trainer_option, train_sampler,
                        train_loader, test_loader,
                        model, scheduler,
                        configs['net']['lossfn'] == 'crf',
                        cls.process_id)

    @classmethod
    def main(cls):
        assert check_argument_types()
        parser = cls.get_parser()
        
        args = parser.parse_args()
        
        ngpus_per_node = torch.cuda.device_count()

        args.world_size = ngpus_per_node * args.world_size

        mp.spawn(cls.main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

def main():
    AsrTraining.main()


if __name__ == "__main__":
    main()