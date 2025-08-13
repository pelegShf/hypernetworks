import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST, KMNIST

DATASETS = {
    "mnist": MNIST,
    "fashion": FashionMNIST,
    "kmnist": KMNIST,
}


def compute_mean_std(dataset,seed=42,n_workers=8):
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=512, shuffle=False, num_workers=n_workers,
        worker_init_fn=lambda _: np.random.seed(seed)
    )
    n_channels = dataset[0][0].shape[0]
    mean = torch.zeros(n_channels)
    std = torch.zeros(n_channels)

    for imgs, _ in loader:
        imgs = imgs.view(imgs.size(0), imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)

    mean /= len(dataset)
    std /= len(dataset)

    return mean, std


def build_loaders(
    data_dir: str,
    batch_size: int,
    tst_batch_size: int,
    workers: int,
    device: torch.device,
    dataset: str = "mnist",
    seed:int = 42
):
    DS = DATASETS[dataset]

    mean, std = compute_mean_std(
        DS(data_dir, train=True, download=True, transform=transforms.ToTensor()),
        seed = seed,
        n_workers=workers
    )
    tfm = transforms.Compose([     
                            transforms.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.9,1.1)), # rotates by +- 10 degs, moves 5%, scales 10%
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)])
    tfm_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)])
    train = DS(data_dir, train=True, download=True, transform=tfm)
    test = DS(data_dir, train=False, download=True, transform=tfm_test)

    pin = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin,
        worker_init_fn=lambda _: np.random.seed(seed)
    )

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=tst_batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=pin,
        worker_init_fn=lambda _: np.random.seed(seed)
    )

    return train_loader, test_loader
