from . import *


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

