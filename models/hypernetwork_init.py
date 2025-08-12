import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os

from functools import reduce
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ExponentialLR

from arguments import get_args
from utils.visual import plot_results


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Get Device: GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}.")
device = torch.device(device)

args = get_args()


torch.manual_seed(args.seed)

train_kwargs = {"batch_size": args.batch_size}
test_kwargs = {"batch_size": args.testing_batch_size}


# Cuda settings
workers = min(8, os.cpu_count() or 4)  # tune

if torch.cuda.is_available():
    cuda_kwargs = {
        "num_workers": workers,
        "pin_memory": True,
        "shuffle": True,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)


# TODO: Calc this live
data_mean = 0.1307
data_std = 0.3081
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((data_mean,), (data_std,))]
)

ds1 = datasets.MNIST("Data/mnist", train=True, download=True, transform=transform)
ds2 = datasets.MNIST("Data/mnist", train=False, transform=transform)


train_loader = torch.utils.data.DataLoader(ds1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(ds2, **test_kwargs)


def init_history():
    return {"train_loss": [], "test_loss": [], "test_acc": []}


# Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = device

        # Array holding dims of hidden layers
        self.layer_dims = [784, 64, 10]
        in_dim, hid_dim, out_dim = self.layer_dims
        self.shapes = {
            "W1": (hid_dim, in_dim),
            "b1": (hid_dim,),
            "W2": (out_dim, hid_dim),
            "b2": (out_dim,),
        }
        self.sizes = {
            k: int(torch.tensor(self.shapes[k]).prod().item()) for k in self.shapes
        }
        self.total_main_net_param = sum(self.sizes.values())

        # self.total_main_net_param = reduce(lambda x, y: x * y, self.layer_dims)

        # Embed vector, predefined and optimized as input to the hypernetwork
        self.embedding = nn.Parameter(
            torch.randn(
                [1, 5], requires_grad=True, dtype=torch.float32, device=self.device
            )
        )

        self.hypernetwork = nn.Sequential(
            nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, self.total_main_net_param)
        )

    def _unpack_flat(self, flat):
        i = 0
        sW1 = self.sizes["W1"]
        W1 = flat[i : i + sW1].view(self.shapes["W1"])
        i += sW1

        sb1 = self.sizes["b1"]
        b1 = flat[i : i + sb1].view(self.shapes["b1"])
        i += sb1

        sW2 = self.sizes["W2"]
        W2 = flat[i : i + sW2].view(self.shapes["W2"])
        i += sW2

        sb2 = self.sizes["b2"]
        b2 = flat[i : i + sb2].view(self.shapes["b2"])
        i += sb2

        return W1, b1, W2, b2

    # def sample_weights(self):
    #     self.hypernet_outputs = self.hypernetwork(self.embedding)[0]
    #     next_idx = 0
    #     for i in range(len(self.layer_dims) - 1):
    #         cur_idx = next_idx
    #         next_idx += self.layer_dims[i] * self.layer_dims[i + 1]

    #         weights_splice = self.hypernet_outputs[cur_idx:next_idx].reshape(
    #             [self.layer_dims[i + 1], self.layer_dims[i]]
    #         )

    #         del self.main_network[i * 2].weight
    #         self.main_network[i * 2].weight = weights_splice

    def forward(self, x):
        x = torch.flatten(x, 1)
        flat = self.hypernetwork(self.embedding)[0]
        W1, b1, W2, b2 = self._unpack_flat(flat)

        h1 = F.linear(x, W1, b1)
        h1 = F.relu(h1)
        out = F.linear(h1, W2, b2)
        return out


def train(model, device, train_loader, optimizer):
    model.train()
    running = 0.0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)

        output = model(data)

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        bsz = data.size(0)
        running += loss.item() * bsz
        total += bsz

    return running / total


def test(model, device, test_loader):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss_sum += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)

    return loss_sum / total, 100.0 * correct / total


model = Net().to(device)


for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name:20}: {reduce(lambda x,y: x*y, param.data.shape)}")


optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

scheduler = ExponentialLR(optimizer, gamma=args.gamma)

hist = init_history()

epoch_times, train_times, test_times = [], [], []


_sync()
total_t0 = time.perf_counter()
for epoch in range(1, args.epochs + 1):
    _sync()
    e0 = time.perf_counter()

    _sync()
    t0 = time.perf_counter()
    tr_loss = train(model, device, train_loader, optimizer)
    _sync()
    t1 = time.perf_counter()
    train_times.append(t1 - t0)

    _sync()
    s0 = time.perf_counter()
    tst_loss, tst_accuracy = test(model, device, test_loader)
    _sync()
    s1 = time.perf_counter()
    test_times.append(s1 - s0)
    scheduler.step()

    _sync()
    e1 = time.perf_counter()
    epoch_times.append(e1 - e0)

    hist["train_loss"].append(tr_loss)
    hist["test_loss"].append(tst_loss)
    hist["test_acc"].append(tst_accuracy)

    print(
        f"Epoch: {epoch:03d} | train_loss {tr_loss:.4f} | test_loss:"
        f"{tst_loss:.4f} | test_accuracy: {tst_accuracy:.4f}"
    )

_sync()
total_time = time.perf_counter() - total_t0
avg_epoch = sum(epoch_times) / len(epoch_times)
avg_train = sum(train_times) / len(train_times)
avg_test = sum(test_times) / len(test_times)

print(
    f"\n⏱ Total: {total_time:.2f}s | "
    f"Avg/epoch: {avg_epoch:.2f}s "
    f"(train {avg_train:.2f}s, test {avg_test:.2f}s)\n"
)
plot_results(hist, "runs/mnist_hypernet")
