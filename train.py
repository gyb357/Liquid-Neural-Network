import torch.nn as nn
import torch.optim as optim
import torch
from torch import device
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Tuple
from tqdm import tqdm


class Trainer():
    def __init__(
            self,
            device: device,
            model: nn.Module,
            loss_fn: nn.Module,
            train_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int = 100,
            batch_size: int = 128,
            lr: float = 0.001,
            tau: float = 0.001
    ) -> None:
        # Model
        self.device = device
        self.model = model.to(device)

        # Data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        # Train modules
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = loss_fn.to(device)

        # Tau
        self.tau = tau

    def _save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def _get_data_ts(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        data = data.squeeze(1)                                                  # Remove channel dimension
        batch_size, seq_len, _ = data.size()                                    # (batch_size, seq_len, in_features)
        ts = torch.ones(batch_size, seq_len, 1, device=self.device) * self.tau  # (batch_size, seq_len)
        return data, ts

    def fit(self) -> None:
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            for (data, labels) in tqdm(self.train_loader, desc=f"[Epoch {epoch}]"):
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                data, ts = self._get_data_ts(data)
                output = self.model(data, ts)
                output = output[:, -1, :]

                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

            # Evaluate the model
            val_acc = self.eval()
            print(f"[Epoch {epoch}] Validation Accuracy: {val_acc:.2f}%")

    def eval(self) -> float:
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for data, labels in tqdm(self.test_loader, desc="[Eval]"):
                data, labels = data.to(self.device), labels.to(self.device)

                data, ts = self._get_data_ts(data)
                output = self.model(data, ts)
                output = output[:, -1, :]
                output = output.argmax(dim=1)
                
                correct += (output == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        return accuracy

