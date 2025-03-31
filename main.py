import torch
import torch.nn as nn
from modules import LTCCell, CfCCell
from model import LNN
from dataset import get_sequential_MNIST_dataloaders, get_sequential_transforms
from train import Trainer


# Model parameters
cell = LTCCell
in_features = 784
hidden_features = 128
out_features = 10
backbone_features = 64 # CfCCell only
backbone_depth = 4 # CfCCell only

# Training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss()
epochs = 100
batch_size = 128
lr = 0.001
tau = 0.001


if __name__ == '__main__':
    # Device
    print(f'device: {device}')

    # Model
    model = LNN(
        cell=cell,
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        backbone_features=backbone_features,
        backbone_depth=backbone_depth
    )
    parameters = model._get_parameters()
    print(f"Number of parameters: {parameters}")

    # Dataset loaders
    train_loader, test_loader = get_sequential_MNIST_dataloaders(batch_size=batch_size, transform=get_sequential_transforms())

    # Trainr
    trainer = Trainer(
        device=device,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        tau=tau
    )
    trainer.fit()
    trainer._save_model(path='model.pth')