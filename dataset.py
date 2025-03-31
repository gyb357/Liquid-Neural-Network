from torchvision import transforms, datasets
from torch.utils.data import DataLoader


@staticmethod
def get_sequential_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

@staticmethod
def get_sequential_MNIST_dataloaders(batch_size: int, transform: transforms.Compose, num_workers: int = 0):
    train_loader = DataLoader(
        dataset=datasets.MNIST(root='./data', train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=datasets.MNIST(root='./data', train=False, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader

