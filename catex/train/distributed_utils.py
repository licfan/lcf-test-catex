import argparse
import dataclasses
from dataclasses import is_dataclass

from catex.utils.build_dataclass import build_dataclass
from typing import check_argument_types

@dataclasses.dataclass
class DistributedOptions:
    workers: int
    rank: int
    dist_url: str
    dist_backend: str
    word_size: int
    gpu: int

class DistributedTrainer:
    def __init__():
        raise RuntimeError("This class can't be instantiated")

    @classmethod
    def build_options(cls, args: argparse.Namespace) -> DistributedOptions:
        assert check_argument_types()
        return build_dataclass(DistributedOptions, args)