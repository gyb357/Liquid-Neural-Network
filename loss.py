import torch.nn as nn
import torch
from torch import Tensor


class MAPELoss(nn.Module):
    def __init__(
            self,
            epsilon: float = 1e-6
    ) -> None:
        super(MAPELoss, self).__init__()
        # Attributes
        self.epsilon = epsilon

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + self.epsilon))) * 100.0

