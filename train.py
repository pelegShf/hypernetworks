from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR

from arguments import get_args
from models.hyper_mlp import HyperMLPGeneric
from Data.build_loaders import build_loaders
from utils.common import set_seed, get_device, cuda_sync
from utils.visual import plot_results


def train_one_epoch(model, device, train_loader, optimizer, clip_grad=None):
    model.train()
    running = 0.0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)

        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        b = data.size(0)
        running += loss.item() * b
        total += b

    return running / total


@torch.no_grad()
def evaluate(model, device, test_loader):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)

        loss_sum += F.cross_entropy(output, target, reduction="sum").item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    return loss_sum / total, 100.0 * correct / total


def train(
    model,
    epochs,
    device,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    clip_grad,
    output_dir,
    verbose=False,
):
    hist = {"train_loss": [], "test_loss": [], "test_acc": []}
    epoch_times, train_times, test_times = [], [], []

    cuda_sync()
    t0_all = time.perf_counter()
    for epoch in range(1, epochs, +1):
        cuda_sync()
        t0 = time.perf_counter()
        tr = train_one_epoch(model, device, train_loader, optimizer, clip_grad)
        cuda_sync()
        t1 = time.perf_counter()

        cuda_sync()
        s0 = time.perf_counter()
        vl, va = evaluate(model, device, test_loader)
        cuda_sync()
        s1 = time.perf_counter()

        scheduler.step()
        epoch_times.append((s1 - t0))
        train_times.append((t1 - t0))
        test_times.append((s1 - s0))

        hist["train_loss"].append(tr)
        hist["test_loss"].append(vl)
        hist["test_acc"].append(va)

        print(f"Epoch {epoch:03d} | train {tr:.4f} | val {vl:.4f} | acc {va:.2f}% ")
    cuda_sync()
    total = time.perf_counter() - t0_all

    if verbose:
        print(
            f"Total {total:.2f}s | avg/epoch {sum(epoch_times)/len(epoch_times):.2f}s "
            f"(train {sum(train_times)/len(train_times):.2f}s, test {sum(test_times)/len(test_times):.2f}s)"
        )

    plot_results(hist, output_dir)


def main():
    args = get_args()
    device = get_device()
    print(f" Using {device}")
    set_seed(args.seed)

    train_loader, test_loader = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        tst_batch_size=args.testing_batch_size,
        workers=args.workers,
        device=device,
        dataset=args.dataset,

    )

    model = HyperMLPGeneric(
        layer_dims=args.layer_dims,
        embed_dim=args.emb_dim,
        use_cnn_cond=args.use_cnn_cond,
    ).to(device)

    if args.verbose:
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"{n:40s} {p.numel():8d}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=args.gamma)
    train(
        model,
        args.epochs,
        device,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        True,
        args.run_dir,
    )


if __name__ == "__main__":
    main()
