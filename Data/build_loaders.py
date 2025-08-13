from __future__ import annotations
import torch
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST, KMNIST

DATASETS = {
    "mnist": MNIST,
    "fashion": FashionMNIST,
    "kmnist": KMNIST,
}


def build_loaders(
    data_dir: str,
    batch_size: int,
    tst_batch_size: int,
    workers: int,
    device: torch.device,
    dataset: str = "mnist",
):
    DS = DATASETS[dataset]

    tfm = transforms.Compose([transforms.ToTensor()])

    train = DS(data_dir, train=True, download=True, transform=tfm)
    test = DS(data_dir, train=False, download=True, transform=tfm)

    pin = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin
    )

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=tst_batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=pin,
    )

    return train_loader, test_loader
