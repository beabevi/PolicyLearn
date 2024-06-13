import torch
from ogb.graphproppred.mol_encoder import AtomEncoder


class Identity(torch.nn.Identity):
    def __init__(self, in_dim, emb_dim, k):
        super().__init__()
        self.k = k


class ZincAtomEncoder(torch.nn.Module):
    def __init__(self, in_dim, emb_dim, k):
        super().__init__()
        self.k = k
        self.enc = torch.nn.Embedding(in_dim, emb_dim - k)

    def forward(self, x):
        return torch.hstack((x[:, : self.k], self.enc(x[:, self.k :].long().squeeze())))


class MyAtomEncoder(torch.nn.Module):
    def __init__(self, in_dim, emb_dim, k):
        super().__init__()
        self.k = k
        self.enc = AtomEncoder(emb_dim - k)

    def forward(self, x):
        return torch.hstack((x[:, : self.k], self.enc(x[:, self.k :].long().squeeze())))


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_dim, emb_dim, k):
        super().__init__()
        self.k = k
        self.enc = torch.nn.Linear(in_dim, emb_dim - k)

    def forward(self, x):
        return torch.hstack((x[:, : self.k], self.enc(x[:, self.k :])))
