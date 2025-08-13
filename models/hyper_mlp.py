import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperMLPGeneric(nn.Module):
    def __init__(
        self, layer_dims, embed_dim=5, use_cnn_cond=True, in_channels=1, dropout=0.2
    ):
        super().__init__()
        assert len(layer_dims) >= 2, "Need at least input and output dims"
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1
        self.emb_dim = embed_dim
        self.use_cnn_cond = use_cnn_cond
        self.dropout = dropout
        ##################################################
        ################ Embedding init ##################
        ################################################## 
        if use_cnn_cond:
            self.cond = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, 1, 1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),  
                nn.Conv2d(32, 64, 3, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, self.emb_dim),
            )
        else:
            in_dim = layer_dims[0]
            self.cond = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, self.emb_dim)
            )
            
        ################################################## 
        ################ Main network init ###############
        ################################################## 
        self.shapes_W, self.shapes_b,sizes = [], [], []
        for i in range(self.L):
            out_d, in_d = layer_dims[i + 1], layer_dims[i]
            self.shapes_W.append((out_d, in_d))
            self.shapes_b.append((out_d,))
            sizes.append(out_d * in_d)
            sizes.append(out_d)
        self.sizes = torch.tensor(sizes, dtype=torch.long)
        self.total = int(self.sizes.sum().item())
       
       ################################################## 
       ################ hypernetwork init ###############
       ################################################## 
        self.hypernetwork = nn.Sequential(
            nn.Linear(self.emb_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, self.total),
        )
        # Make sure the output for the first time it small as it is the weight of
        # the main network.
        with torch.no_grad():
            self.hypernetwork[-1].weight.mul_(0.01)
            self.hypernetwork[-1].bias.mul_(0.01)


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
        x_flat = x.view(B, -1) if x.dim() == 4 else x

        # Gets the embeding
        if self.use_cnn_cond:
            assert x.dim() == 4, "CNN expects [B,C,H,W]"
            z = self.cond(x)
        else:
            z = self.cond(x_flat)

        # Gets the weights from the hypernetwork
        flat = self.hypernetwork(z)
        Ws, bs = self._unpack_batched(flat)

        h = x_flat
        for i in range(self.L):
            # h is [B,d_in] and Ws[i] is [B,d_out,d_in] -> h: [B,1,d_in] & Ws[i]: [B,d_in,d_out]
            mm = torch.bmm(h.unsqueeze(1), Ws[i].transpose(1, 2))
            # bs[i] is [B, d_out] -> [B,1,d_out]
            h = (mm + bs[i].unsqueeze(1)).squeeze(1)
            if i < self.L - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
