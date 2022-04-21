import torch

class FFModule(torch.nn.Module):
    """Feed-forward module

    default output dimension = idim
    x0 -> LayerNorm -> FC -> Swish -> Dropout -> FC -> Dropout -> x1
    x0 + res_factor * x1 -> output
    """

    def __init__(self, idim: int, res_factor: float = 0.5, dropout: float = 0.0) -> None:
        super().__init__()
        assert res_factor > 0. and dropout >= 0.
        self._res_factor = res_factor

        self.ln = torch.nn.LayerNorm([idim])
        self.fc0 = torch.nn.Linear(idim, idim*4)
        self.swish = torch.nn.SiLU()
        self.dropout0 = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(idim*4, idim)
        self.dropout1 = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        output = self.ln(x)
        output = self.fc0(output)
        output = self.swish(output)
        output = self.dropout0(output)
        output = self.fc1(output)
        output = self.dropout1(output)
        output = x + self._res_factor * output

        return output
