import os.path as osp
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from scipy.special import comb
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data.data import Data
from torch_geometric.datasets import ZINC
from torch_geometric.datasets import TUDataset as TUDataset_
from torch_geometric.transforms import Constant


class GraphCountDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = osp.join(root, name)
        self.url = "https://github.com/LingxiaoShawn/GNNAsKernel/raw/main/data/subgraphcount/raw/randomgraph.mat"
        super().__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # Normalize as in GNN-AK
        self.data.y = self.data.y / self.data.y.std(0)

        a = sio.loadmat(self.raw_paths[0])
        self.train_idx = torch.from_numpy(a["train_idx"][0])
        self.val_idx = torch.from_numpy(a["val_idx"][0])
        self.test_idx = torch.from_numpy(a["test_idx"][0])

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        a = sio.loadmat(self.raw_paths[0])

        A = a["A"][0]

        data_list = []
        for i in range(len(A)):
            a = A[i]
            A2 = a.dot(a)
            A3 = A2.dot(a)
            tri = np.trace(A3) / 6
            tailed = ((np.diag(A3) / 2) * (a.sum(0) - 2)).sum()
            cyc4 = 1 / 8 * (np.trace(A3.dot(a)) + np.trace(A2) - 2 * A2.sum())
            cus = a.dot(np.diag(np.exp(-a.dot(a).sum(1)))).dot(a).sum()

            deg = a.sum(0)
            star = 0
            for j in range(a.shape[0]):
                star += comb(int(deg[j]), 3)

            expy = torch.tensor([[tri, tailed, star, cyc4, cus]])

            E = np.where(A[i] > 0)
            edge_index = torch.Tensor(np.vstack((E[0], E[1]))).type(torch.int64)
            x = torch.ones(A[i].shape[0], 1)
            data_list.append(Data(edge_index=edge_index, x=x, y=expy.to(torch.float)))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def separate_data(self):
        return {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}


class TUDataset(TUDataset_):
    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_node_attr: bool = False,
        use_edge_attr: bool = False,
        cleaned: bool = False,
    ):
        super().__init__(
            root,
            name,
            transform,
            pre_transform,
            pre_filter,
            use_node_attr,
            use_edge_attr,
            cleaned,
        )
        self.mean, self.std = None, None

    def download(self):
        super().download()

    def process(self):
        super().process()

    def separate_data(self, seed, fold_idx):
        # code taken from GIN and adapted
        # since we only consider train and valid, use valid as test

        assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        labels = self.data.y.numpy()
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, test_idx = idx_list[fold_idx]

        return {
            "train": torch.tensor(train_idx),
            "val": torch.tensor(test_idx),
            "test": torch.tensor(test_idx),
        }

    def set_alchemy_mean_std(self, mean, std):
        self.mean = mean
        self.std = std


def get_data(
    dataset_name: str,
    dataroot: Path,
    split: int = 0,
) -> Tuple[InMemoryDataset, InMemoryDataset, InMemoryDataset]:
    if dataset_name == "graphcount":
        dataset = GraphCountDataset(dataroot, dataset_name)
        dataset.data.y = dataset.data.y[:, split : split + 1]
        split_idx = dataset.separate_data()
        train_dataset, val_dataset, test_dataset = (
            dataset[split_idx["train"]],
            dataset[split_idx["val"]],
            dataset[split_idx["test"]],
        )

    elif dataset_name == "ZINC":
        train_dataset = ZINC(dataroot / dataset_name, subset=True, split="train")
        val_dataset = ZINC(dataroot / dataset_name, subset=True, split="val")
        test_dataset = ZINC(dataroot / dataset_name, subset=True, split="test")

    elif "ogbg" in dataset_name:
        dataset = PygGraphPropPredDataset(root=dataroot, name=dataset_name)
        split_idx = dataset.get_idx_split()
        train_dataset, val_dataset, test_dataset = (
            dataset[split_idx["train"]],
            dataset[split_idx["valid"]],
            dataset[split_idx["test"]],
        )
    elif dataset_name == "REDDIT-BINARY":
        dataset = TUDataset(root=dataroot, name=dataset_name, transform=Constant())
        split_idx = dataset.separate_data(seed=42, fold_idx=split)
        train_dataset, val_dataset, test_dataset = (
            dataset[split_idx["train"]],
            dataset[split_idx["val"]],
            dataset[split_idx["test"]],
        )
    elif dataset_name == "alchemy_full":

        def f(i):
            return pd.read_csv(i, header=None).to_numpy().squeeze(0)

        indices = np.concatenate(
            list(
                map(
                    f,
                    [
                        "data/alchemy_indices/train_al_10.index",
                        "data/alchemy_indices/val_al_10.index",
                        "data/alchemy_indices/test_al_10.index",
                    ],
                )
            )
        )

        dataset = TUDataset(root=dataroot, name="alchemy_full")[indices]

        # Normalize as in SignNet
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std

        dataset.set_alchemy_mean_std(mean, std)
        train_dataset, val_dataset, test_dataset = (
            dataset[:10_000],
            dataset[10_000:11_000],
            dataset[11_000:12_000],
        )
    else:
        raise ValueError(f"dataset {dataset_name} not supported")

    return train_dataset, val_dataset, test_dataset
