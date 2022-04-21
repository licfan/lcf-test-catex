import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Sequence

def pad_list(
            xs: torch.Tensor, 
            pad_value: float = 0.0,
            dim: int = 0,
            ) -> torch.Tensor:
    """
    Perform padding for the list of tensors

    Args:
        xs : torch.Tensor
        pad_value: float, Value for padding

    Returns:
        Tensor: Padded tensor (B, TMax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor(1., 1., 1., 1.), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    
    """
    if dim == 0:
        return pad_sequence(xs, 
                            batch_first=True,
                            padding_value = pad_value)

    else:
        xs = [x.tranpose(0, dim) for x in xs]
        padded = pad_sequence(xs, 
                            batch_first=True, 
                            padding_value=pad_value)

        return padded.transpose(1, dim+1).contiguous()


def str2num(src: str) -> Sequence[int]:
    return list(src.encode())

def num2str(num_list: list) -> str:
    return bytes(num_list).decode()
