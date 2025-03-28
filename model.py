import torch.nn as nn
import torch
from typing import Callable, Union
from modules import LTCCell, CFCCell
from torch import Tensor


class LNN(nn.Module):
    def __init__(
            self,
            cell: Callable[..., Union[LTCCell, CFCCell]],
            in_features: int,
            hidden_features: int,
            out_features: int,
            dt: float = 0.01,
            backbone_depth: int = 1
    ) -> None:
        super(LNN, self).__init__()
        # Cell layer
        if cell == LTCCell:
            self.cell = cell(in_features, hidden_features)
        if cell == CFCCell:
            self.cell = cell(in_features, hidden_features, backbone_depth)

        # Outout layer
        self.out = nn.Linear(hidden_features, out_features)

        # Attributes
        self.dt = dt

    def _get_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: Tensor):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.cell.hidden_features, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            h = self.cell(x_t, h, self.dt)

        out = self.out(h)
        return out

