import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR

from arguments import get_args
from models.hyper_mlp import HyperMLPGeneric
from Data.build_loaders import build_loaders
from utils.common import set_seed, get_device
from utils.visual import plot_results



def train_one_epoch(model, device, train_loader, optimizer, clip_grad=None):
    model.train()
    running,correct, total = 0.0, 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)

        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1)

        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        b = data.size(0)
        correct += (pred == target).sum().item()
        running += loss.item() * b
        total += b

    return running / total, 100 * (correct / total)


@torch.no_grad()
def evaluate(model, device, test_loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1)

        b = data.size(0)
        loss_sum += loss.item() * b
        correct += (pred == target).sum().item()
        total += b

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
    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, epochs + 1):
        tr, ta = train_one_epoch(model, device, train_loader, optimizer, clip_grad)

        vl, va = evaluate(model, device, test_loader)


        scheduler.step()

        hist["train_loss"].append(tr)
        hist["train_acc"].append(ta)
        hist["test_loss"].append(vl)
        hist["test_acc"].append(va)

        print(f"Epoch {epoch:03d} | train {tr:.4f} - {ta:.4f}% | test {vl:.4f} - {va:.2f}% ")
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
        seed = args.seed
    )

    model = HyperMLPGeneric(
        layer_dims=args.layer_dims,
        embed_dim=args.emb_dim,
        use_cnn_cond=args.use_cnn_cond,
        dropout=args.dropout,
    ).to(device)

    if args.verbose:
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"{n:40s} {p.numel():8d}")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = ExponentialLR(optimizer=optimizer, gamma=args.gamma)
    train(
        model,
        args.epochs,
        device,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        clip_grad=args.clip_grad,
        output_dir=args.run_dir + args.dataset,
    )


if __name__ == "__main__":
    main()
