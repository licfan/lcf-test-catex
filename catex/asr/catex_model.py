import torch

from catex.asr.specaug import SpecAug

from typeguard import check_argument_types
from typeguard import check_return_type

class CatexModel(torch.nn.Module):
    "CRF  based acoustic model"
    def __init__(
        self, 
        am: torch.nn.Module,
        criterion: torch.nn.Module,
        specaug: SpecAug = None
    ):
        super().__init__()
        self.infer = am
        self.criterion = criterion
        self.specaug = specaug

    def forward(
        self,
        logits,
        labels,
        input_lengths,
        label_lengths
    ):
        assert check_argument_types()
        if self.specaug is not None and self.training:
            logits, input_lengths = self.specaug(logits, input_lengths)

        labels = labels.cpu()
        input_lengths = input_lengths.cpu()
        label_lengths = label_lengths.cpu()

        netout, lens_o = self.infer(logits, input_lengths)
        netout = torch.log_softmax(netout, dim = -1)

        loss = self.criterion(
                            netout, 
                            labels, 
                            lens_o.to(torch.int32).cpu(),
                            label_lengths)
        
        return loss