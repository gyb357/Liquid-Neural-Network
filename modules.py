import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class LTCCell(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            activation: nn.Module = nn.Tanh
    ) -> None:
        """
        Args:
            in_features (int): number of input features
            hidden_features (int): number of hidden features
            activation (nn.Module): activation function to use (default: Tanh)
        """

        super(LTCCell, self).__init__()
        # Attributes
        self.hidden_features = hidden_features
        self.activation = activation()
        self.epsilon = 1e-6

        # Weights
        self.w_i = nn.Parameter(torch.Tensor(hidden_features, in_features))     # Input
        self.w_h = nn.Parameter(torch.Tensor(hidden_features, hidden_features)) # Hidden

        # Biases
        self.b = nn.Parameter(torch.Tensor(hidden_features))     # bias
        self.b_tau = nn.Parameter(torch.Tensor(hidden_features)) # τ
        self.b_A = nn.Parameter(torch.Tensor(hidden_features))   # A

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.w_i)
        nn.init.xavier_uniform_(self.w_h)
        nn.init.zeros_(self.b)
        nn.init.zeros_(self.b_tau)
        nn.init.ones_(self.b_A)

    def forward(self, x: Tensor, h: Optional[Tensor] = None, t: Optional[Tensor] = None) -> Tensor:
        batch_size = x.size(0)

        # Initialize hidden state and time span
        if h is None:
            h = torch.zeros(batch_size, self.hidden_features, device=x.device)
        if t is None:
            t = torch.ones(batch_size, device=x.device)

        # Reshape time span
        t = t.view(batch_size, 1)

        # f = f(x(t), I(t), t, θ)
        f = self.activation(F.linear(x, self.w_i) + F.linear(h, self.w_h) + self.b)

        # τ Must be positive (softplus)
        tau = F.softplus(self.b_tau) + self.epsilon

        # Fused step
        # x(t + ∆t) = (x(t) + ∆t * f ⊙ A) / (1 + ∆t * (1 / τ + f))
        h = (h + t * f * self.b_A) / (1 + t * (1 / tau + f))
        return h


class CfCCell(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            backbone_features: int = 4,
            backbone_depth: int = 4,
            activation: nn.Module = nn.Tanh
    ) -> None:
        """
        Args:
            in_features (int): number of input features
            hidden_features (int): number of hidden features
            backbone_features (int): number of features in the backbone (default: 4)
            backbone_depth (int): number of layers in the backbone (default: 4)
            activation (nn.Module): activation function to use (default: Tanh)
        """

        super(CfCCell, self).__init__()
        # Attributes
        self.hidden_features = hidden_features
        self.activation = activation()
        self.epsilon = 1e-6

        # Weights
        self.w_i = nn.Parameter(torch.Tensor(backbone_features, in_features + hidden_features))
        self.w_b = nn.ParameterList([
            nn.Parameter(torch.Tensor(backbone_features, backbone_features))
            for _ in range(backbone_depth - 1)
        ])
        self.w_f = nn.Parameter(torch.Tensor(hidden_features, backbone_features))

        # Biases
        self.b_i = nn.Parameter(torch.Tensor(backbone_features))
        self.b_b = nn.ParameterList([
            nn.Parameter(torch.Tensor(backbone_features))
            for _ in range(backbone_depth - 1)
        ])
        self.b_f = nn.Parameter(torch.Tensor(hidden_features))
        self.b_tau = nn.Parameter(torch.Tensor(hidden_features)) # τ
        self.b_A = nn.Parameter(torch.Tensor(hidden_features))   # A

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.w_i)
        nn.init.zeros_(self.b_i)

        for w, b in zip(self.w_b, self.b_b):
            nn.init.xavier_uniform_(w)
            nn.init.zeros_(b)

        nn.init.xavier_uniform_(self.w_f)
        nn.init.zeros_(self.b_f)
        nn.init.zeros_(self.b_tau)
        nn.init.ones_(self.b_A)

    def forward(self, x: Tensor, h: Optional[Tensor] = None, t: Optional[Tensor] = None) -> Tensor:
        batch_size = x.size(0)

        # Initialize hidden state and time span
        if h is None:
            h = torch.zeros(batch_size, self.hidden_features, device=x.device)
        if t is None:
            t = torch.ones(batch_size, device=x.device)

        # Reshape time span
        t = t.view(batch_size, 1)

        # Concatenate
        x = torch.cat([x, h], dim=1)

        # Backbone
        backbone = self.activation(F.linear(x, self.w_i, self.b_i))

        for w, b in zip(self.w_b, self.b_b):
            backbone = self.activation(F.linear(backbone, w, b))

        # f = f(I(t), θ)
        f = F.linear(backbone, self.w_f, self.b_f)

        # f and τ Must be positive (softplus)
        f = F.softplus(f) + self.epsilon
        tau = F.softplus(self.b_tau) + self.epsilon

        # x(t) ≈ (x0 - A) * e(-t * (τ + f)) * f + A
        h = -self.b_A * torch.exp(-t * (tau + f)) * f + self.b_A
        return h


class CfCImprovedCell(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            backbone_features: int = 4,
            backbone_depth: int = 4,
            activation: nn.Module = nn.Tanh
    ) -> None:
        """
        Args:
            in_features (int): number of input features
            hidden_features (int): number of hidden features
            backbone_features (int): number of features in the backbone (default: 4)
            backbone_depth (int): number of layers in the backbone (default: 4)
            activation (nn.Module): activation function to use (default: Tanh)
        """

        super(CfCImprovedCell, self).__init__()
        # Attributes
        self.hidden_features = hidden_features
        self.activation = activation()
        self.epsilon = 1e-6

        # Weights
        self.w_i = nn.Parameter(torch.Tensor(backbone_features, in_features + hidden_features))
        self.w_b = nn.ParameterList([
            nn.Parameter(torch.Tensor(backbone_features, backbone_features))
            for _ in range(backbone_depth - 1)
        ])

        # Biases
        self.b_i = nn.Parameter(torch.Tensor(backbone_features))
        self.b_b = nn.ParameterList([
            nn.Parameter(torch.Tensor(backbone_features))
            for _ in range(backbone_depth - 1)
        ])

        # Head weights & biases
        self.w_g = nn.Parameter(torch.Tensor(hidden_features, backbone_features))
        self.w_f = nn.Parameter(torch.Tensor(hidden_features, backbone_features))
        self.w_h = nn.Parameter(torch.Tensor(hidden_features, backbone_features))
        self.w_tau = nn.Parameter(torch.Tensor(hidden_features, backbone_features))
        self.b_g = nn.Parameter(torch.Tensor(hidden_features))
        self.b_f = nn.Parameter(torch.Tensor(hidden_features))
        self.b_h = nn.Parameter(torch.Tensor(hidden_features))
        self.b_tau = nn.Parameter(torch.Tensor(hidden_features))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.w_i)
        nn.init.zeros_(self.b_i)

        for w, b in zip(self.w_b, self.b_b):
            nn.init.xavier_uniform_(w)
            nn.init.zeros_(b)

        for w, b in [(self.w_g, self.b_g), (self.w_f, self.b_f), (self.w_h, self.b_h), (self.w_tau, self.b_tau)]:
            nn.init.xavier_uniform_(w)
            nn.init.zeros_(b)

    def forward(self, x: Tensor, h: Optional[Tensor] = None, t: Optional[Tensor] = None) -> Tensor:
        batch_size = x.size(0)

        # Initialize hidden state and time span
        if h is None:
            h = torch.zeros(batch_size, self.hidden_features, device=x.device)
        if t is None:
            t = torch.ones(batch_size, device=x.device)

        # Reshape time span
        t = t.view(batch_size, 1)

        # Concatenate
        x = torch.cat([x, h], dim=1)

        # Backbone
        backbone = self.activation(F.linear(x, self.w_i, self.b_i))

        for w, b in zip(self.w_b, self.b_b):
            backbone = self.activation(F.linear(backbone, w, b))

        # Head gates
        head_g = torch.tanh(F.linear(backbone, self.w_g, self.b_g))
        head_f = F.linear(backbone, self.w_f, self.b_f)
        head_h = torch.tanh(F.linear(backbone, self.w_h, self.b_h))
        tau = F.linear(backbone, self.w_tau, self.b_tau)

        sigma = torch.sigmoid((tau + head_f) * t)
        h = head_h * (1 - sigma) + head_g * sigma
        return h


class RNNCell(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            activation: nn.Module = nn.Tanh
    ) -> None:
        """
        Args:
            in_features (int): number of input features
            hidden_features (int): number of hidden features
            activation (nn.Module): activation function to use (default: Tanh)
        """

        super(RNNCell, self).__init__()
        # Attributes
        self.hidden_features = hidden_features
        self.activation = activation()

        # Weights
        self.w_i = nn.Parameter(torch.Tensor(hidden_features, in_features))
        self.w_h = nn.Parameter(torch.Tensor(hidden_features, hidden_features))

        # Biases
        self.b_i = nn.Parameter(torch.Tensor(hidden_features))
        self.b_h = nn.Parameter(torch.Tensor(hidden_features))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        stdv = 1.0 / (self.hidden_features ** 0.5)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        batch_size = x.size(0)

        # Initialize hidden state
        if h is None:
            h = torch.zeros(batch_size, self.hidden_features, device=x.device)

        # Linear transformations
        h = F.linear(x, self.w_i, self.b_i) + F.linear(h, self.w_h, self.b_h)

        # Apply activation function
        h = self.activation(h)
        return h


class LSTMCell(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int
    ) -> None:
        """
        Args:
            in_features (int): number of input features
            hidden_features (int): number of hidden features
            activation (nn.Module): activation function to use (default: Tanh)
        """

        super(LSTMCell, self).__init__()
        # Attributes
        self.hidden_features = hidden_features

        # Weights
        self.w_i = nn.Parameter(torch.Tensor(4 * hidden_features, in_features))
        self.w_h = nn.Parameter(torch.Tensor(4 * hidden_features, hidden_features))

        # Biases
        self.b = nn.Parameter(torch.Tensor(4 * hidden_features))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        stdv = 1.0 / (self.hidden_features ** 0.5)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

        with torch.no_grad():
            self.b[self.hidden_features:2 * self.hidden_features].fill_(1.0)

    def forward(self, x: Tensor, h: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        batch_size = x.size(0)

        # Initialize hidden state
        if h is None:
            h = torch.zeros(batch_size, self.hidden_features, device=x.device)
            c = torch.zeros(batch_size, self.hidden_features, device=x.device)
        else:
            h, c = h

        # Linear transformations
        gates = F.linear(x, self.w_i) + F.linear(h, self.w_h) + self.b

        # Split into four gates
        i, f, g, o = gates.chunk(4, dim=1)

        # Apply activation functions
        i = torch.sigmoid(i) # Input gate
        f = torch.sigmoid(f) # Forget gate
        g = torch.tanh(g)    # Candidate cell state
        o = torch.sigmoid(o) # Output gate

        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class GRUCell(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int
    ) -> None:
        """
        Args:
            in_features (int): number of input features
            hidden_features (int): number of hidden features
            activation (nn.Module): activation function to use (default: Tanh)
        """
        
        super(GRUCell, self).__init__()
        # Attributes
        self.hidden_features = hidden_features

        # Weights (input)
        self.w_ir = nn.Parameter(torch.Tensor(hidden_features, in_features)) # Reset gate
        self.w_iz = nn.Parameter(torch.Tensor(hidden_features, in_features)) # Update gate
        self.w_in = nn.Parameter(torch.Tensor(hidden_features, in_features)) # New gate

        # Weights (hidden state)
        self.w_hr = nn.Parameter(torch.Tensor(hidden_features, hidden_features)) # Reset gate
        self.w_hz = nn.Parameter(torch.Tensor(hidden_features, hidden_features)) # Update gate
        self.w_hn = nn.Parameter(torch.Tensor(hidden_features, hidden_features)) # New gate

        # Biases
        self.b_ir = nn.Parameter(torch.Tensor(hidden_features))
        self.b_iz = nn.Parameter(torch.Tensor(hidden_features))
        self.b_in = nn.Parameter(torch.Tensor(hidden_features))
        self.b_hr = nn.Parameter(torch.Tensor(hidden_features))
        self.b_hz = nn.Parameter(torch.Tensor(hidden_features))
        self.b_hn = nn.Parameter(torch.Tensor(hidden_features))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        stdv = 1.0 / (self.hidden_features ** 0.5)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

        with torch.no_grad():
            self.b_iz.fill_(1.0)
            self.b_hz.fill_(1.0)

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        batch_size = x.size(0)

        # Initialize hidden state
        if h is None:
            h = torch.zeros(batch_size, self.hidden_features, device=x.device)

        # Linear transformations
        r = torch.sigmoid(F.linear(x, self.w_ir, self.b_ir) + F.linear(h, self.w_hr, self.b_hr))  # Reset gate
        z = torch.sigmoid(F.linear(x, self.w_iz, self.b_iz) + F.linear(h, self.w_hz, self.b_hz))  # Update gate
        n = torch.tanh(F.linear(x, self.w_in, self.b_in) + r * F.linear(h, self.w_hn, self.b_hn)) # New gate

        h = (1 - z) * n + z * h
        return h

