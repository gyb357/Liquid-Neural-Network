import torch.nn as nn
import torch
from typing import Callable, Union
from modules import LTCCell, CfCCell, CfCImprovedCell
from torch import Tensor


class LNN(nn.Module):
    def __init__(
            self,
            cell: Callable[..., Union[LTCCell, CfCCell, CfCImprovedCell]],
            in_features: int,
            hidden_features: int,
            out_features: int,
            backbone_features: int = 1,
            backbone_depth: int = 1
    ) -> None:
        super(LNN, self).__init__()
        if cell == LTCCell:
            self.cell = LTCCell(in_features, hidden_features)
        elif cell in [CfCCell, CfCImprovedCell]:
            self.cell = cell(in_features, hidden_features, backbone_features=backbone_features, backbone_depth=backbone_depth)

        # Output layer
        self.out = nn.Linear(hidden_features, out_features)

        # Attributes
        self.hidden_features = hidden_features

    def _get_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor, ts: Tensor) -> Tensor:
        # x:   Input
        # ts:  Timespans
        # h:   Hidden state
        # out: Output

        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_features, device=x.device)

        # RNN loop
        outputs = []
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h, ts[:, t])
            outputs.append(self.out(h))

        # Output layer
        out = torch.stack(outputs, dim=1)
        return out


class LNNEnsemble(nn.Module):
    def __init__(
        self,
        base_model: LNN,
        ensemble_size: int = 2
    ) -> None:
        super(LNNEnsemble, self).__init__()
        self.models = nn.ModuleList([base_model() for _ in range(ensemble_size)])

    def _get_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor, ts: Tensor) -> Tensor:
        outputs = [model(x, ts) for model in self.models]
        return torch.stack(outputs, dim=0).mean(dim=0)


class LSTM(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int,
            layer_depth: int = 1,
    ) -> None:
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(in_features, hidden_features, layer_depth, batch_first=True)
        self.out = nn.Linear(hidden_features, out_features)
        
        # Attributes
        self.hidden_features = hidden_features

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if isinstance(p, nn.LSTM):
                for name, param in p.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(p, nn.Linear):
                nn.init.xavier_uniform_(p.weight)
                nn.init.zeros_(p.bias)

    def _get_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: Tensor, _: Tensor) -> Tensor:
        # x:   Input
        # h0:  Initial hidden state
        # c0:  Initial cell state
        # out: Output

        batch_size, _, _ = x.size()
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_features, device=x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_features, device=x.device)

        # Output layer
        out, _ = self.lstm(x, (h0, c0))
        out = self.out(out)
        return out

