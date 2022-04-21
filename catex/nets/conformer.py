import torch

from catex.nets.feed_forward import FFModule
from catex.nets.attention import MHSAModule
from catex.nets.positional_encoding import PositionalEncoding

from catex.nets.convolution import Conv2dSubsampling
from catex.nets.convolution import ConvModule

from collections import OrderedDict

class ConformerCell(torch.nn.Module):
    def __init__(
            self,
            idim: int,
            res_factor: float = 0.5,
            d_head: int = 36,
            num_heads: int = 4,
            kernel_size: int = 32,
            multiplier: int = 1,
            dropout: float = 0.1):
        super().__init__()

        self.ffm0 = FFModule(idim, res_factor, dropout)
        self.mhsam = MHSAModule(idim, d_head, num_heads, dropout)
        self.convm = ConvModule(
            idim, kernel_size, dropout, multiplier)
        self.ffm1 = FFModule(idim, res_factor, dropout)
        self.ln = torch.nn.LayerNorm(idim)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):

        ffm0_out = self.ffm0(x)
        attn_out, attn_ls = self.mhsam(ffm0_out, lens)
        conv_out = self.convm(attn_out)
        ffm1_out = self.ffm1(conv_out)
        out = self.ln(ffm1_out)
        return out, attn_ls


class ConformerNet(torch.nn.Module):
    """The conformer model with convolution subsampling

    Args:
        num_cells (int): number of conformer blocks
        idim (int): dimension of input features
        hdim (int): hidden size in conformer blocks
        num_classes (int): number of output classes
        conv_multiplier (int): the multiplier to conv subsampling module
        dropout_in (float): the dropout rate to input of conformer blocks (after
                   the linear and subsampling layers)
        res_factor(float): the weighted-factor or residual-connected  shortcut in
                         feed-forward module

        d_head(int): dimension of heads in multi-head attention module
        num_heads(int): number of heads in multi-head attention module
        kernel_size(int): kernel_size in convolution module
        multiplier(int): multiplier of depth conv in convolution module
        dropout (float): dropout rate to all conformer internal modules
        delta_feats (bool): True if the input features contains delta and delta-delta features
    """
    def __init__(
        self,
        num_cells: int,
        idim: int,
        hdim: int,
        num_classes: int,
        conv_multiplier: int,
        dropout_in: float = 0.2,
        res_factor: float = 0.5,
        d_head: int = 36,
        num_heads: int = 4,
        kernel_size: int = 32,
        multiplier: int = 1,
        dropout: float = 0.1,
        delta_feats = False
    ):
        super().__init__()

        if delta_feats:
            idim = idim // 3
        
        self.conv_subsampling = Conv2dSubsampling(
            conv_multiplier, stackup=delta_feats)
        self.linear_drop = torch.nn.Sequential(OrderedDict({
            'linear': torch.nn.Linear((idim // 4) * conv_multiplier, hdim),
            'dropout': torch.nn.Dropout(dropout_in)
        }))
        self.cells = torch.nn.ModuleList()
        pe = PositionalEncoding(hdim)
        for i in range(num_cells):
            cell = ConformerCell(
                hdim, res_factor, d_head, num_heads, kernel_size, multiplier, dropout)
            self.cells.append(cell)
            # FIXME: Note that this is somewhat hard-code style
            cell.mhsam.mha.pe = pe
        self.classifier = torch.nn.Linear(hdim, num_classes)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        x_subsampled, ls_subsampled = self.conv_subsampling(x, lens)
        out = self.linear_drop(x_subsampled)
        ls = ls_subsampled
        for cell in self.cells:
            out, ls = cell(out, ls)
        logits = self.classifier(out)

        return logits, ls