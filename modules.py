import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor


class LTCCell(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int
    ) -> None:
        super(LTCCell, self).__init__()
        # f(x(t), I(t), t, θ)
        self.w_i = nn.Linear(in_features, hidden_features, bias=False)
        self.w_h = nn.Linear(hidden_features, hidden_features, bias=False)

        # Parameters
        self.bias = nn.Parameter(torch.zeros(hidden_features))
        self.tau = nn.Parameter(torch.ones(hidden_features))
        self.A = nn.Parameter(torch.zeros(hidden_features))

        # Attributes
        self.hidden_features = hidden_features

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x: Tensor, h: Tensor, t: Tensor) -> Tensor:
        # f = f(x(t), I(t), t, θ)
        f = F.tanh(self.w_i(x) + self.w_h(h) + self.bias)

        # τ Must be positive (softplus)
        tau = F.softplus(self.tau) + 1e-8 # epsilon for stability

        # FusedStep
        # x(t + ∆t) = (x(t) + ∆t * f(x(t), I(t), t, θ) ⊙ A) / (1 + ∆t * (1 / τ + f(x(t), I(t), t, θ)))
        h_new = (h + t * f * self.A) / (1 + t * (1 / tau + f)) # FusedStep
        return h_new


class CFCCell(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            backbone_depth: int = 1
    ) -> None:
        super(CFCCell, self).__init__()
        # Input layer
        layers = [nn.Linear(in_features + hidden_features, hidden_features, bias=False)]

        # Backbone layers
        for _ in range(backbone_depth):
            layers.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features, bias=False),
                nn.BatchNorm1d(hidden_features),
                nn.LeakyReLU(inplace=True),
            ))
        self.backbone = nn.Sequential(*layers)

        # Neural network heads
        self.head_g = nn.Linear(hidden_features, hidden_features)
        self.head_f = nn.Linear(hidden_features, hidden_features)
        self.head_h = nn.Linear(hidden_features, hidden_features)
        self.tau = nn.Linear(hidden_features, hidden_features)

        # Attributes
        self.hidden_features = hidden_features

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.xavier_uniform_(p)

    def forward(self, x: Tensor, h: Tensor, t: Tensor) -> Tensor:
        # Ensure t is a tensor with the correct shape
        if not isinstance(t, Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
            t = t.expand(x.size(0), 1)

        # Concatenate time step
        t = t.view(x.size(0), 1)
        x = torch.cat([x, h], dim=1)

        # Backbone layers
        x = self.backbone(x)

        # Neural network heads
        head_g = F.tanh(self.head_g(x))
        head_f = self.head_f(x)
        head_h = F.tanh(self.head_h(x))

        # Approximation of τ
        tau = self.tau(x)

        # Advanced numerical DE solvers
        # x(t) = σ * (-f(x, I, θ_f) * t) ⊙ g(x, I, θ_g) + [1 - σ * (-[f(x, I, θ_f)] * t)] ⊙ h(x, I, θ_h)
        sigma = F.sigmoid((tau + head_f) * t)
        h_new = head_h * (1 - sigma) + head_g * sigma
        return h_new

