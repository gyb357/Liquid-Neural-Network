from . import *


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

