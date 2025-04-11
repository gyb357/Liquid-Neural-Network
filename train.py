import torch.nn as nn
# import torch.optim as optim
# import torch
# import matplotlib.pyplot as plt
from torch import device
from dataset import KRXDataLoader
# from loss import MAPELoss
# from tqdm import tqdm
# from typing import Tuple
# from torch.utils.data import DataLoader


class Trainer():
    def __init__(
            self,
            device: device,
            model: nn.Module,
            dataloader: KRXDataLoader,
            
    )


# class Trainer():
#     def __init__(
#             self,
#             device: device,
#             model: nn.Module,
#             dataloader: KRXDataLoader,
#             epochs: int,
#             seq_len: int,
#             lr: float,
#             do_lnn: bool,
#             tau: float,
#             plot: bool = False,
#             save: bool = True
#     ) -> None:
#         # Attributes
#         self.device = device
#         self.model = model
#         self.epochs = epochs
#         self.seq_len = seq_len
#         self.do_lnn = do_lnn
#         self.tau = tau
#         self.plot = plot
#         self.save = save
        
#         # Train modules
#         self.optimizer = optim.Adam(model.parameters(), lr=lr)
#         self.criterion = nn.MSELoss()
#         self.metric = MAPELoss()

#         # Data loaders
#         self.train_loader, self.val_loader, self.test_loader = dataloader.get_dataloader()

#     def save_model(self, path: str) -> None:
#         torch.save(self.model.state_dict(), path)

#     def fit(self) -> None:
#         for epoch in range(self.epochs):
#             self.model.train()
#             loss, metrics = 0, 0

#             for x, y in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch"):
#                 x, y = x.to(self.device), y.to(self.device)
#                 y = y.squeeze(1)

#                 self.optimizer.zero_grad()

#                 if self.do_lnn:
#                     batch_size = x.size(0)
#                     ts = torch.ones(batch_size, self.seq_len, 1, device=x.device) * self.tau
#                     outputs = self.model(x, ts)
#                 else:
#                     outputs = self.model(x)

#                 loss = self.criterion(outputs, y)
#                 metric = self.metric(outputs, y)

#                 loss.backward()
#                 self.optimizer.step()

#                 loss += loss.item()
#                 metrics += metric.item()

#             loss /= len(self.train_loader)
#             metrics /= len(self.train_loader)
#             print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss:.4f}, Metric: {metrics:.4f}")

#             val_loss, val_metrics = self.eval(self.val_loader)
#             print(f"Validation Loss: {val_loss:.4f}, Metric: {val_metrics:.4f}")

#         test_loss, test_metrics = self.eval(self.test_loader)
#         print(f"Test Loss: {test_loss:.4f}, Metric: {test_metrics:.4f}")
#         print("Training complete.")

#         # Plotting predictions
#         if self.plot:



#             plt.figure(figsize=(14, 6))
#             plt.plot(test_dates, trues_denorm.numpy(), label='True Price', color='black')
#             plt.plot(test_dates, preds_denorm.numpy(), label='Predicted Price', color='red', linestyle='--')
#             plt.title("Predicted vs True Close Price (Test Set)")
#             plt.xlabel("Date")
#             plt.ylabel("Price")
#             plt.legend()
#             plt.grid(True)
#             plt.tight_layout()
#             plt.show()

#         # Save the model
#         if self.save:
#             self.save_model(self.save_path)
#             print(f"Model saved to {self.save_path}")

#     def eval(self, dataloader: DataLoader) -> Tuple[float, float]:
#         self.model.eval()
#         loss, metrics = 0, 0

#         with torch.no_grad():
#             for x, y in dataloader:
#                 x, y = x.to(self.device), y.to(self.device)

#                 y = y.squeeze(1)
#                 if self.do_lnn:
#                     batch_size = x.size(0)
#                     ts = torch.ones(batch_size, self.seq_len, 1, device=x.device) * self.tau
#                     outputs = self.model(x, ts)
#                 else:
#                     outputs = self.model(x)

#                 loss += self.criterion(outputs, y).item()
#                 metric += self.metric(outputs, y).item()

#         loss /= len(self.val_loader)
#         metrics /= len(self.val_loader)
#         return loss, metrics

