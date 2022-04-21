import torch

from torch.nn import ConstantPad2d
from torch.nn import Conv2d
from torch.nn import Conv1d
from torch.nn import LayerNorm
from torch.nn import ReLU
from torch.nn import GLU
from torch.nn import Conv1d
from torch.nn import BatchNorm1d
from torch.nn import SiLU
from torch.nn import Dropout

from catex.nets.stack import StackDelta

class SiLU(torch.nn.Module):
    @staticmethod
    def forward(x):
        return x*torch.sigmoid(x)

class ConvModule(torch.nn.Module):
    def __init__(self,
                idim: int,
                kernel_size: int = 32,
                dropout: float = 0.,
                multiplier: int = 1):
        super().__init__()

        self.ln = LayerNorm([idim])
        self.pointwise_conv0 = Conv1d(
                    idim, 2 * idim,
                     kernel_size=1, stride=1)

        self.glu = GLU(dim=1)

        cdim = idim

        padding = (kernel_size-1)//2

        self.padding = ConstantPad2d(
            (padding, kernel_size-1-padding, 0, 0), 0.)

        self.depthwise_conv = Conv1d(
                cdim, multiplier*cdim,
                kernel_size=kernel_size,
                 stride=1, groups=cdim,
                  padding=0)
                  
        cdim = multiplier * cdim
        self.bn = BatchNorm1d(cdim)
        self.swish = SiLU()
        self.pointwise_conv1 = Conv1d(cdim, idim, kernel_size=1, stride=1)
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # [B, T, D]
        output = self.ln(x)
        # [B, T, D] -> [B, D, T]
        output = output.transpose(1, 2)
        # [B, D, T] -> [B, 2*D, T]
        output = self.pointwise_conv0(output)
        # [B, 2D, T] -> [B, D, T]
        output = self.glu(output)
        # [B, D, T] -> [B, multiplier*D, T]
        output = self.padding(output)
        output = self.depthwise_conv(output)
        if output.size(0) > 1 or output.size(2) > 1:
            # Doing batchnorm with [1, D, 1] tensor raise error.
            output = self.bn(output)
        output = self.swish(output)
        # [B, multiplier*D, T] -> [B, D, T]
        output = self.pointwise_conv1(output)
        output = self.dropout(output)
        # [B, D, T] -> [B, T, D]
        output = output.transpose(1, 2)

        return x + output

class Conv2dSubsampling(torch.nn.Module):
    """
    Conv2dSubsampling: From Google TensorflowASR
https://github.com/TensorSpeech/TensorFlowASR/blob/5fbd6a89b93b703888662f5c47d05bae256e98b0/tensorflow_asr/models/layers/subsampling.py
Originally wrote with Tensorflow, translated into PyTorch by Huahuan.
    
    """
    def __init__(self, multiplier: int, stackup: bool = True):
            super().__init__()
            self._lens_in_args_ = None
            def _unsqueeze(x): return x.unsqueeze(1)

            if stackup:
                self.stack = StackDelta()
                idim = 3
            else:
                self.stack = _unsqueeze
                idim = 1
            #NOTE (wwx): ReLU->swish
            self.conv = torch.nn.Sequential(
                ConstantPad2d(padding=(0, 1, 0, 1), value=0.),
                Conv2d(idim, multiplier, kernel_size=3, stride=2),
                ReLU(),
                ConstantPad2d(padding=(0, 1, 0, 1), value=0.),
                Conv2d(multiplier, multiplier, kernel_size=3, stride=2),
                ReLU()
            )

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4 (not strictly).
        """
        # [B, T, D] -> [B, C, T, D]
        x = self.stack(x)
        # [B, C, T, D] -> [B, OD, T//4, D//4]
        out = self.conv(x)
        B, OD, NT, ND = out.size()
        # [B, OD, T//4, D//4] -> [B, T//4, OD, D//4]
        out = out.permute(0, 2, 1, 3)
        # [B, T//4, OD, D//4] -> [B, T//4, OD * D//4]
        out = out.contiguous().view(B, NT, OD*ND)
        # NOTE (Huahuan): use torch.div() instead '//'
        # NOTE (wwx): torch.version<1.6,use floor_divide instead //
        if torch.__version__ >= '1.8.0':
            lens_out = torch.div(lens, 2, rounding_mode='floor')
            lens_out = torch.div(lens_out, 2, rounding_mode='floor')
        else:
            lens_out = torch.floor_divide(lens, 2)
            lens_out = torch.floor_divide(lens_out, 2)
            
        return out, lens_out




