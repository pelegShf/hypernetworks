from __future__ import annotations
from typing import Tuple
import torch
from torchvision import datasets, transforms


def build_loaders(
    data_dir: str,
    batch_size: int,
    tst_batch_size: int,
    workers: int,
    device: torch.device,
    normalize: bool = False,
):
    tfm = transforms.Compose([transforms.ToTensor()])

    train = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)

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
