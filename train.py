import torch.nn as nn
import torch.optim as optim
import torch
from torch import device
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer():
    def __init__(
            self,
            device: device,
            model: nn.Module,
            train_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int = 100,
            batch_size: int = 128,
            lr: float = 0.01
    ) -> None:
        # Device
        self.device = device
        # Model
        self.model = model
        # Data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader
        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        # Train modules
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def eval(self, data_loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                preds = output.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total * 100
        return acc

    def train(self) -> None:
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (x, y) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

            train_acc = correct / total * 100
            avg_loss = total_loss / len(self.train_loader)
            print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%")

            val_acc = self.eval(self.test_loader)
            print(f"Test Accuracy: {val_acc:.2f}%")

        # Model save
        torch.save(self.model.state_dict(), 'model.pth')

