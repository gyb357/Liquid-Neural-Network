from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(784, 1))
    ])

def get_dataloaders(batch_size: int, transforms: transforms, num_workers: int = 0) -> dict:
    train_loader = DataLoader(
        dataset=datasets.MNIST(root='./data', train=True, download=True, transform=transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=datasets.MNIST(root='./data', train=False, download=True, transform=transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader

