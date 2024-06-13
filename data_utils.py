import abc
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Optional

import torch
import torch_scatter
from torch.profiler import record_function
from torch.utils.data import DataLoader, default_collate
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

import debug
from extensions import repeat_interleave, vrange
from observation import Observation, Tensor1D, Tensor2D


def lengths2slices(lengths: torch.LongTensor) -> torch.LongTensor:
    return torch.hstack(
        (
            torch.tensor(0, dtype=torch.int64, device=lengths.device),
            lengths.cumsum(-1),
        )
    )


# num_graphs: int = V.nnz(0)
# num_total_subgraphs: int = V.nnz(0, 1)
# num_total_nodes: int = V.nnz(0, 1, 2)
# num_original_nodes: (num_graphs,) = V.nnz(0, 2)


@dataclass(eq=False)
class AdjAndFeats:
    """
    Representation of
    V: (g, subg, node, feat)
    E: (g, subg, node, node, feat)
                 ^^^^^^^^^^
                 local_edge_indices in COO format
    """

    num_subgraphs_per_graph: Tensor1D["num_graphs"]

    num_nodes_per_subgraph: Tensor1D["num_total_subgraphs"]
    v_features: Tensor2D["num_total_nodes", "v_d"]

    num_edges_per_subgraph: Tensor1D["num_total_subgraphs"]
    local_edge_indices: Tensor2D[2, "num_total_edges"]
    e_features: Optional[Tensor2D["num_total_edges", "e_d"]]

    ys: Tensor2D["num_graphs", "y_d"]
    subgraph_ids: Optional[Tensor1D["num_graphs"]] = None

    def to(self, dev) -> "AdjAndFeats":
        return AdjAndFeats(
            num_subgraphs_per_graph=self.num_subgraphs_per_graph.to(dev),
            num_nodes_per_subgraph=self.num_nodes_per_subgraph.to(dev),
            v_features=self.v_features.to(dev),
            num_edges_per_subgraph=self.num_edges_per_subgraph.to(dev),
            local_edge_indices=self.local_edge_indices.to(dev),
            e_features=None if self.e_features is None else self.e_features.to(dev),
            ys=self.ys.to(dev),
            subgraph_ids=(
                None if self.subgraph_ids is None else self.subgraph_ids.to(dev)
            ),
        )

    def replace(self, v_features):
        assert self.v_features.size(0) == v_features.size(0)
        res = AdjAndFeats(
            num_subgraphs_per_graph=self.num_subgraphs_per_graph,
            num_nodes_per_subgraph=self.num_nodes_per_subgraph,
            v_features=v_features,
            num_edges_per_subgraph=self.num_edges_per_subgraph,
            local_edge_indices=self.local_edge_indices,
            e_features=self.e_features,
            ys=self.ys,
            subgraph_ids=self.subgraph_ids,
        )
        # NOTE: Big WARNING: here we propagate all (possibly cached) properties
        #  since changing only "v_features" doesn't change  the rest.
        #  If a new cached_property is added or data-dependencies
        #  change this must be updated!!
        for k, v in self.__dict__.items():
            if k != "v_features":
                res.__dict__[k] = v
        return res

    def construct_egonets(self, marking, num_hops):
        assert self.v_features.size(0) == marking.size(0)

        (roots,) = marking.sum(-1).nonzero(as_tuple=True)

        # Construct the ego networks around the marked nodes
        _, _, _, edge_mask = k_hop_subgraph(
            roots,
            num_hops,
            self.flat_edge_index,
            relabel_nodes=False,
            num_nodes=self.num_nodes_per_graph.sum(),
        )

        e_indices = indices_from_ranges_for_gather(
            self.e_subg_slices, self.g_slices[:-1]
        )
        if debug.DEBUG:
            # NOTE(first_node_id_is_1): 0 subgraph is the graph itself
            # (with no marked nodes) so ensure its edges were not previously included
            assert not edge_mask[e_indices].any().item()

        # NOTE(first_node_id_is_1): 0 subgraph is the graph itself
        # (with no marked nodes) thus include its edges, which were masked out
        # by `k_hop_subgraph`
        edge_mask[e_indices] = True

        local_edge_indices = self.local_edge_indices[:, edge_mask]
        e_features = (
            self.e_features[edge_mask]
            if self.e_features is not None
            else self.e_features
        )
        num_edges_per_subgraph = torch_scatter.segment_csr(
            src=edge_mask.int(), indptr=self.e_subg_slices, reduce="sum"
        )
        if debug.DEBUG:
            # NOTE(first_node_id_is_1): 0 subgraph is the graph itself
            # so the number of edges of the 0 subgraph before and after creating
            # the egonets must be the snae
            assert (
                self.num_edges_per_subgraph[self.g_slices[:-1]]
                == num_edges_per_subgraph[self.g_slices[:-1]]
            ).all()

        res = AdjAndFeats(
            num_subgraphs_per_graph=self.num_subgraphs_per_graph,
            num_nodes_per_subgraph=self.num_nodes_per_subgraph,
            v_features=self.v_features,
            num_edges_per_subgraph=num_edges_per_subgraph,
            local_edge_indices=local_edge_indices,
            e_features=e_features,
            ys=self.ys,
            subgraph_ids=self.subgraph_ids,
        )
        return res

    @property
    def device(self):
        return self.num_subgraphs_per_graph.device

    @property
    def num_graphs(self) -> int:
        return self.num_subgraphs_per_graph.size(0)

    @property
    def num_features(self) -> int:
        return self.v_features.size(-1)

    @property
    def num_total_subgraphs(self) -> int:
        return self.num_nodes_per_subgraph.size(0)

    @cached_property
    def g_slices(self) -> Tensor1D["num_graphs + 1"]:
        return lengths2slices(self.num_subgraphs_per_graph)

    @cached_property
    def v_subg_slices(self) -> Tensor1D["num_total_subgraphs + 1"]:
        return lengths2slices(self.num_nodes_per_subgraph)

    @cached_property
    def e_subg_slices(self) -> Tensor1D["num_total_subgraphs + 1"]:
        return lengths2slices(self.num_edges_per_subgraph)

    @record_function("AdjAndFeats::at")
    def at(self, which_graphs, subgraphs, subgraph_slices) -> "AdjAndFeats":
        num_subgraphs_per_graph = subgraph_slices[1:] - subgraph_slices[:-1]
        broadcast_graphs = repeat_interleave(which_graphs, num_subgraphs_per_graph)

        if debug.DEBUG:
            assert (subgraphs < self.num_subgraphs_per_graph[broadcast_graphs]).all()

        start_of_subgraphs = self.g_slices[broadcast_graphs]

        global_subgraph_ids = start_of_subgraphs + subgraphs

        num_nodes_per_subgraph = self.num_nodes_per_subgraph[global_subgraph_ids]
        v_indices = indices_from_ranges_for_gather(
            self.v_subg_slices, global_subgraph_ids
        )
        v_features = self.v_features[v_indices]

        num_edges_per_subgraph = self.num_edges_per_subgraph[global_subgraph_ids]
        e_indices = indices_from_ranges_for_gather(
            self.e_subg_slices, global_subgraph_ids
        )
        local_edge_indices = self.local_edge_indices[:, e_indices]
        e_features = None if self.e_features is None else self.e_features[e_indices]

        return AdjAndFeats(
            num_subgraphs_per_graph=num_subgraphs_per_graph,
            num_nodes_per_subgraph=num_nodes_per_subgraph,
            v_features=v_features,
            num_edges_per_subgraph=num_edges_per_subgraph,
            local_edge_indices=local_edge_indices,
            e_features=e_features,
            ys=self.ys[which_graphs],
            subgraph_ids=subgraphs,
        )

    @cached_property
    def num_original_nodes_per_graph(self):  # -> X.nnz(0, 2)
        # NOTE assumes that all subgraphs have the same number of nodes
        num_original_nodes_per_graph: Tensor1D[self.num_graphs,] = (
            self.num_nodes_per_subgraph[self.g_slices[:-1]]
        )
        return num_original_nodes_per_graph

    @cached_property
    def num_nodes_per_graph(self) -> int:
        return torch_scatter.scatter_add(
            self.num_nodes_per_subgraph, self.subgraph2graph
        )

    @cached_property
    def original_node2graph(self):  # -> ? : Shape[X.nnz(0, 2).size(0),]
        # Index `g` of V[:, 0] : (g, node, feat)
        return repeat_interleave(
            torch.arange(self.num_graphs, device=self.device),
            self.num_original_nodes_per_graph,
        )

    @cached_property
    def node2graph(self):  # -> ? : Shape[X.nnz(0, 2).size(0),]
        # Index `g` of V[:, 0] : (g, subg, node, feat)
        return repeat_interleave(
            torch.arange(self.num_graphs, device=self.device),
            self.num_nodes_per_graph,
        )

    @cached_property
    def original_g_v_slices(self):
        return lengths2slices(self.num_original_nodes_per_graph)

    @cached_property
    def global_node_id(self):  # -> ? : Shape[self.num_total_nodes,]
        # Index of V.flatten(g, subg, node)
        starts = self.original_g_v_slices[:-1]
        return vrange(self.num_nodes_per_subgraph, starts[self.subgraph2graph])

    @cached_property
    def subgraph2graph(self):  # -> ? : Shape[self.num_subgraphs,]
        return repeat_interleave(
            torch.arange(self.num_graphs, device=self.device),
            self.num_subgraphs_per_graph,
        )

    @cached_property
    def node2subgraph(self):  # -> ? : Shape[self.num_total_nodes,]
        return repeat_interleave(
            vrange(self.num_subgraphs_per_graph),
            self.num_nodes_per_subgraph,
        )

    @cached_property
    def first_edge_index_and_attr(self):  # -> E[:, 0]
        # NOTE: Equivalent to
        #   self.at(
        #     torch.arange(self.num_graphs),
        #     torch.zeros(self.num_graphs, dtype=torch.long),
        #     torch.arange(self.num_graphs + 1)
        #   ).flat_edge_index
        e_indices = indices_from_ranges_for_gather(
            self.e_subg_slices, self.g_slices[:-1]
        )
        local_edge_indices = self.local_edge_indices[:, e_indices]
        nodes_before: Tensor1D[self.num_graphs] = self.original_g_v_slices[:-1]
        per_edge_offset = repeat_interleave(
            nodes_before, self.num_edges_per_subgraph[self.g_slices[:-1]]
        )
        edge_attr = None if self.e_features is None else self.e_features[e_indices]
        return (per_edge_offset + local_edge_indices), edge_attr

    @cached_property
    def flat_edge_index(self):  # -> Shape[2, self.num_total_edges]
        # Index after we flatten (g, subg, node, node)
        nodes_before: Tensor1D[self.num_total_subgraphs] = self.v_subg_slices[:-1]
        per_edge_offset = repeat_interleave(nodes_before, self.num_edges_per_subgraph)
        return per_edge_offset + self.local_edge_indices


def indices_from_ranges_for_gather(slices, start_indices):
    start_of_elements = slices[start_indices]
    end_of_elements = slices[start_indices + 1]
    num_elements_per_subgraph = end_of_elements - start_of_elements
    indices = vrange(num_elements_per_subgraph, start_of_elements)
    return indices


def sample_subgraphs(
    dataset: AdjAndFeats, graph_id: torch.LongTensor, max_subgraphs: int, size: int
) -> Observation:
    size = size if size is not None else max_subgraphs
    graph_id = graph_id.view((-1, 1))
    num_subg_per_graph = dataset.num_subgraphs_per_graph[graph_id].to(graph_id.device)
    probs = torch.rand(
        (graph_id.size(0), num_subg_per_graph.max()),
        device=num_subg_per_graph.device,
    )
    indices = torch.broadcast_to(
        torch.arange(num_subg_per_graph.max(), device=probs.device), probs.shape
    )
    probs[:, 0] = 1.0  # NOTE(first_node_id_is_1): 0 subgraph is the graph itself
    probs[indices >= num_subg_per_graph] = -torch.inf

    max_subg = torch.minimum(
        torch.tensor(max_subgraphs, device=num_subg_per_graph.device),
        num_subg_per_graph,
    )
    k = torch.ceil(max_subg).to(torch.long)

    subgraphs = torch.topk(probs, min(size, probs.size(-1)))[1]
    subgraphs = torch.cat(
        (
            subgraphs,
            torch.full(
                (subgraphs.size(0), max(0, size - probs.size(-1))),
                fill_value=-1,
                device=subgraphs.device,
            ),
        ),
        dim=-1,
    )
    indices = torch.broadcast_to(
        torch.arange(size, device=probs.device), subgraphs.shape
    )
    subgraphs[indices >= k] = -1

    return Observation(
        graph_id=graph_id,
        which_subgraphs=subgraphs,
        ys=dataset.ys[graph_id.squeeze()],
    )


def dataset2tensor(dataset: InMemoryDataset, dev=None, num_marked=3) -> AdjAndFeats:
    num_subgraphs_per_graph = []
    num_nodes_per_subgraph = []
    num_edges_per_subgraph = []

    v_features = []
    local_edge_indices = []
    e_features = []
    ys = []

    for data in tqdm(dataset, desc="Processing..", mininterval=5, leave=False):
        num_subgraphs = data.num_nodes + 1
        for subgraph_idx in range(0, num_subgraphs):
            root_node = None if subgraph_idx == 0 else subgraph_idx - 1
            subgraph_data = get_subgraph_node_marking(
                data, num_marked=num_marked, root_node=root_node
            )
            num_nodes_per_subgraph.append(subgraph_data.num_nodes)
            num_edges_per_subgraph.append(subgraph_data.num_edges)

            v_features.append(subgraph_data.x)
            local_edge_indices.append(subgraph_data.edge_index)
            if subgraph_data.edge_attr is not None:
                e_features.append(subgraph_data.edge_attr)

        ys.append(data.y)
        num_subgraphs_per_graph.append(num_subgraphs)

    return AdjAndFeats(
        torch.tensor(num_subgraphs_per_graph, device=dev),
        torch.tensor(num_nodes_per_subgraph, device=dev),
        torch.vstack(v_features).to(device=dev),
        torch.tensor(num_edges_per_subgraph, device=dev),
        torch.hstack(local_edge_indices).to(device=dev),
        None if len(e_features) == 0 else torch.cat(e_features, dim=0).to(device=dev),
        torch.vstack(ys).to(device=dev),
    )


def get_subgraph_node_marking(
    data: Data, num_marked: int, root_node: Optional[int]
) -> Data:
    ids = torch.zeros(num_marked).repeat(data.num_nodes, 1)
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    if root_node is not None:
        ids[root_node] = 1

    x = torch.hstack([ids, data.x]) if data.x is not None else ids.to(torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=data.num_nodes,
    )


class CustomCollateFn:
    def __init__(self, dataset: AdjAndFeats, num_subgraphs: int) -> None:
        self.dataset = dataset
        self.num_subgraphs = num_subgraphs

    def __call__(self, graph_indices):
        graph_indices = default_collate(graph_indices)
        obs = sample_subgraphs(self.dataset, graph_indices, 1, self.num_subgraphs)
        return obs


def mk_collate_fn(dataset: AdjAndFeats, num_subgraphs):
    return CustomCollateFn(dataset=dataset, num_subgraphs=num_subgraphs)


def adj_and_feats_dataloader(
    dataset: AdjAndFeats,
    batch_size: int,
    collate_fn,
    shuffle: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
):
    return DataLoader(
        torch.arange(dataset.num_graphs, device=dataset.device),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def _get_linear_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
