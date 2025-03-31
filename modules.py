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
        self.tau = nn.Parameter(torch.zeros(hidden_features))
        self.A = nn.Parameter(torch.ones(hidden_features))

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
        tau = F.softplus(self.tau) + 1e-8  # epsilon for stability

        # Fused Step
        # x(t + ∆t) = (x(t) + ∆t * f ⊙ A) / (1 + ∆t * (1 / τ + f))
        h_new = (h + t * f * self.A) / (1 + t * (1 / tau + f))
        return h_new


class CfCCell(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            backbone_features: int = 1,
            backbone_depth: int = 1
    ) -> None:
        super(CfCCell, self).__init__()
        # Activation function
        activation = nn.Tanh

        # Input layer
        input_layer = nn.ModuleList([
            nn.Linear(in_features + hidden_features, backbone_features),
            activation()
        ])

        # Backbone layers
        self.backbone = nn.ModuleList([
            nn.Linear(backbone_features, backbone_features),
            activation()
        ] * (backbone_depth - 1))
        self.backbone = nn.Sequential(*input_layer, *self.backbone)

        # Linear layer for f
        self.f = nn.Linear(backbone_features, hidden_features)

        # Parameters
        self.tau = torch.nn.Parameter(torch.zeros(1, hidden_features))
        self.A = torch.nn.Parameter(torch.ones(1, hidden_features))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.xavier_uniform_(p)

    def forward(self, x: Tensor, h: Tensor, t: Tensor) -> Tensor:
        t = t.view(x.size(0), 1)
        x = torch.cat([x, h], 1)

        # Backbone
        x = self.backbone(x)

        # f = f(I(t), θ)
        f = self.f(x)
        
        # x(t) ≈ (x0 - A) * e(-t * (τ + f)) * f + A
        h_new = -self.A * torch.exp(-t * (torch.abs(self.tau) + torch.abs(f))) * f + self.A
        return h_new

