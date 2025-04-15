from . import *


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

