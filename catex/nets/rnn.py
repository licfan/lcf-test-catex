import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import numpy as np
from catex.nets.vgg import VGG2L
import torch.nn.functional as F



def get_vgg2l_odim(idim, in_channel = 1, out_channel = 128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype = np.float32) / 2) # 1st max pooling
    idim = np.ceil(np.array(idim, dtype = np.float32) / 2) # 2nd max pooing
    return int(idim) * out_channel # number of channels

#TODO: disable _LSTM
class _LSTM(torch.nn.Module):
    def __init__(
                self,
                idim: int,
                hdim: int,
                n_layers: int,
                dropout: float = 0.0,
                bidirectional: bool = False):
        super(_LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
            idim,
            hdim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout
        )
    
    def forward(self,
                x: torch.Tensor,
                ilens: torch.Tensor,
                hidden: bool = None):
        self.lstm.flatten_parameters()

        packed_input = pack_padded_sequence(
            x, ilens.to("cpu"),
            batch_first = True
        )

        packed_output, _ = self.lstm(packed_input, hidden)
        out, olens = pad_packed_sequence(
                            packed_output, 
                            batch_first=True)

        return out, olens


class LSTM(torch.nn.Module):
    def __init__(self,
                 idim: int,
                 hdim: int,
                 n_layers, int,
                 num_classes: int,
                 dropout: float,
                 bidirectional: bool = False):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
                                  idim, 
                                  hdim, 
                                  num_layers = n_layers,
                                  bidirectional = bidirectional,
                                  batch_first = True,
                                  dropout = dropout)
        #TODO: disable _LSTM
        #self.lstm = _LSTM(idim,
        #                  hdim,
        #                  n_layers,
        #                  dropout,
        #                  bidirectional=bidirectional)
        if bidirectional:
            self.linear = torch.nn.Linear(hdim * 2, num_classes)
        else:
            self.linear = torch.nn.Linear(hdim, num_classes)


    def forward(self,
                x: torch.Tensor,
                ilens: torch.Tensor,
                hidden = None):
        
        lstm_out, olens = self.lstm(x, ilens, hidden)
        out = self.lstm(lstm_out)
        return out, olens

class BLSTM(LSTM):
    def __init__(self,
                 idim,
                 hdim,
                 n_layers,
                 num_classes,
                 dropout):
        super().__init__(idim,
                         hdim, 
                         n_layers, 
                         num_classes, 
                         dropout, 
                         bidirectional = True)

class VGGLSTM(LSTM):
    def __init__(self,
                 idim,
                 hdim,
                 n_layers,
                 num_classes,
                 dropout,
                 in_channel=3,
                 bidirectional=False):
        super().__init__(get_vgg2l_odim(idim,
                                        in_channel = in_channel),
                                        hdim,
                                        n_layers,
                                        num_classes,
                                        dropout,
                                        bidirectional=bidirectional)

        self.VGG = VGG2L(in_channel)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor):
        vgg_o , vgg_lens = self.VGG(x, ilens)
        return super().forward(vgg_o, vgg_lens)


class VGGBLSTM(VGGLSTM):
    def __init__(self, idim, hdim, n_layers, num_classes, dropout, in_channel=3):
        super().__init__(idim, hdim, n_layers, num_classes,
                         dropout, in_channel=in_channel, bidirectional=True)



class Lookahead(torch.nn.Module):
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = torch.nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1,
                              groups=self.n_features, padding=0, bias=None)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        return x


class LSTMrowCONV(torch.nn.Module):
    def __init__(self, idim, hdim, n_layers, num_classes, dropout):
        super().__init__()

        self.lstm = _LSTM(idim, hdim, n_layers, dropout)
        self.lookahead = Lookahead(hdim, context=5)
        self.linear = torch.nn.Linear(hdim, num_classes)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor, hidden=None):
        lstm_out, olens = self.lstm(x, ilens, hidden)
        ahead_out = self.lookahead(lstm_out)
        return self.linear(ahead_out), olens
