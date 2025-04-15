from . import *


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

