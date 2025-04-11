import torch.nn as nn
import torch
from typing import Union, Optional
from modules import LTCCell, CfCCell, CfCImprovedCell, RNNCell, LSTMCell, GRUCell
from torch import Tensor


class LNN(nn.Module):
    def __init__(
            self,
            cell: Union[LTCCell, CfCCell, CfCImprovedCell],
            in_features: int,
            hidden_features: int,
            out_features: int,
            backbone_features: int = 4,
            backbone_depth: int = 4
    ) -> None:
        """
        Args:
            cell (Union[LTCCell, CfCCell, CfCImprovedCell]): The cell type to use.
            in_features (int): number of input features
            hidden_features (int): number of hidden features
            backbone_features (int): number of features in the backbone (default: 4)
            backbone_depth (int): number of layers in the backbone (default: 4)
            activation (nn.Module): activation function to use (default: Tanh)
        """
        
        super(LNN, self).__init__()
        # Attributes
        self.hidden_features = hidden_features

        # Input layer
        if cell == LTCCell:
            self.cell = cell(in_features, hidden_features)
        elif cell in [CfCCell, CfCImprovedCell]:
            self.cell = cell(in_features, hidden_features, backbone_features, backbone_depth)
        else:
            raise ValueError("cell must be 'LTCCell', 'CfCCell' or 'CfCImprovedCell'")
        
        # Output layer
        self.out = nn.Linear(hidden_features, out_features)

    def _get_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: Tensor, ts: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.size()

        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_features, device=x.device)

        outputs = x.new_empty(batch_size, seq_len, self.out.out_features)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h, ts[:, t])
            outputs[:, t, :] = self.out(h)
        return outputs[:, -1, :]


class RNN(nn.Module):
    def __init__(
            self,
            cell: Union[RNNCell, LSTMCell, GRUCell],
            in_features: int,
            hidden_features: int,
            out_features: int
    ) -> None:
        """
        Args:
            cell (Union[RNNCell, LSTMCell, GRUCell]): The cell type to use.
            in_features (int): number of input features
            hidden_features (int): number of hidden features
            out_features (int): number of output features
        """
        
        super(RNN, self).__init__()
        # Attributes
        self.hidden_features = hidden_features

        # Input layer
        if self.cell not in [RNNCell, LSTMCell, GRUCell]:
            raise ValueError("cell must be 'RNNCell', 'LSTMCell' or 'GRUCell'")
        else:
            self.cell = cell(in_features, hidden_features)

        # Output layer
        self.out = nn.Linear(hidden_features, out_features)

        # Check if the cell is LSTM
        self.is_lstm = isinstance(self.cell, LSTMCell)

    def _get_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.size()

        # Initialize hidden state
        if self.is_lstm:
            if h is None:
                h = torch.zeros(batch_size, self.hidden_features, device=x.device)
                c = torch.zeros(batch_size, self.hidden_features, device=x.device)
            else:
                h, c = h
        else:
            if h is None:
                h = torch.zeros(batch_size, self.hidden_features, device=x.device)

        outputs = x.new_empty(batch_size, seq_len, self.out.out_features)
        for t in range(seq_len):
            if self.is_lstm:
                h, c = self.cell(x[:, t, :], (h, c))
            else:
                h = self.cell(x[:, t, :], h)
            outputs[:, t, :] = self.out(h)
        return outputs[:, -1, :]

