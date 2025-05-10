from . import *


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

