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


class HyperMLPGeneric(nn.Module):
    def __init__(self, layer_dims, embed_dim=5, use_cnn_cond=True):
        super().__init__()
        assert len(layer_dims) >= 2, "Need at least input and output dims"
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1
        self.emb_dim = embed_dim

        if use_cnn_cond:
            self.cond = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 2, 1),
                nn.ReLU(),  # 28->14
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.ReLU(),  # 14 -> 7
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, self.emb_dim),
            )
        else:
            in_dim = layer_dims[0]
            self.cond = nn.Sequential(
                nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(64, self.emb_dim)
            )

        self.shapes_W = []
        self.shapes_b = []
        sizes = []
        for i in range(self.L):
            out_d, in_d = layer_dims[i + 1], layer_dims[i]
            self.shapes_W.append((out_d, in_d))
            self.shapes_b.append((out_d,))
            sizes.append(out_d * in_d)
            sizes.append(out_d)
        self.sizes = torch.tensor(sizes, dtype=torch.long)
        self.total = int(self.sizes.sum().item())

        self.hypernetwork = nn.Sequential(
            nn.Linear(self.emb_dim, 64), nn.ReLU(), nn.Linear(64, self.total)
        )

    def _unpack_batched(self, flat):
        B = flat.size(0)
        pieces = torch.split(flat, self.sizes.tolist(), dim=1)
        Ws, bs = [], []
        p = 0
        for i in range(self.L):
            W = pieces[p].view(B, *self.shapes_W[i])
            p += 1
            b = pieces[p].view(B, *self.shapes_b[i])
            p += 1
            Ws.append(W)
            bs.append(b)
        return Ws, bs

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 4:
            z = self.cond(x)
            x_flat = torch.flatten(x, 1)
        else:
            x_flat = x
            z = self.cond(x_flat)
        flat = self.hypernetwork(z)
        Ws, bs = self._unpack_batched(flat)

        h = x_flat
        for i in range(self.L):
            h = torch.baddbmm(
                bs[i].unsqueeze(1), h.unsqueeze(1), Ws[i].transpose(1, 2)
            ).squeeze(1)
            if i < self.L - 1:
                h = F.relu(h)
                h = F.dropout(h, p=0.2, training=self.training)
        return h
