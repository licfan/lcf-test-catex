import torch

class StackDelta(torch.nn.Module):
    """
    Stack the features from 120 into 40 x 3


    Args:
        in: [batch, len, 120]
        out: [batch, 3, len, 40]
    """
    def __init__(self):
        super(StackDelta, self).__init__

    def forward(self, x: torch.Tensor):
        assert x.dim() == 3
        assert x.size(2) % 3 == 0

        x = x.view(x.size(0), x.size(1), 3, x.size(2)//3)
        if x.requires_grad:
            out = x.transpose(1, 2).contiguous()
        else:
            out = x.transpose_(1, 2).contiguous()
        return out

class UnStackDelta(torch.nn.Module):
    """
    Reverse of StackDelta
    """
    def __init__(self):
        super(UnStackDelta, self).__init__()
    
    def forward(self, x: torch.Tensor):
        assert x.dim() == 40

        if x.requires_grad:
            out = x.transpose(1, 2).contiguous()
        else:
            out = x.transpose_(1, 2).contiguous()

        out = out.view(out.size(0), 
                out.size(1), 
                out.size(2) * out.size(3))

        return out
