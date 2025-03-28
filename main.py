import torch
from modules import LTCCell, CFCCell
from model import LNN
from mnist import get_transforms, get_dataloaders
from train import Trainer


# Model parameters
cell = CFCCell
in_features = 1
hidden_features = 128
out_features = 10

# LTCCell specific
dt = 0.01
# CFCCell specific
backbone_depth = 8

# Training parameters
epochs = 50
batch_size = 128
lr = 0.01


if __name__ == '__main__':
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = LNN(
        cell=cell,
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        dt=dt,
        backbone_depth=backbone_depth
    ).to(device)
    parameters = model._get_parameters()
    print(f"Number of parameters: {parameters}")

    # Dataset loaders
    train_loader, test_loader = get_dataloaders(batch_size=batch_size, transforms=get_transforms())

    # Train
    trainer = Trainer(
        device=device,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )
    trainer.train()

