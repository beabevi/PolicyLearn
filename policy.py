from contextlib import contextmanager
from typing import Tuple

import numpy as np
import torch
import torch_scatter

import debug
from data_utils import AdjAndFeats
from extensions import repeat_interleave, vrange
from models import DSnetwork
from observation import (
    Observation,
    Tensor1D,
    Tensor2D,
    observation_coos,
    update_observation_inplace_no_replace,
)
from utils import top_k_gumbel_softmax


class GumbelSelection(torch.nn.Module):
    def __init__(
        self,
        num_marked: int,
        num_conv_steps: int,
        dataset: AdjAndFeats,
        features_extractor,
        tau: float,
        num_subgraphs: int,
        drop_ratio: float,
        downstream_emb_dim: int,
    ):
        super().__init__()
        self.num_marked = num_marked
        self._dataset = dataset
        self.features_extractor = features_extractor
        self.tau = tau
        self.num_subgraphs = num_subgraphs
        self.drop_ratio = drop_ratio

        self.probs = []
        self.indices = []

        if self.features_extractor is not None:
            emb_dim = self.features_extractor.backbone.emb_dim
            self.node_selector = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=emb_dim,
                    out_features=2 * emb_dim,
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 * emb_dim, out_features=1),
            )

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        raise AttributeError(
            "Can't explicitly set dataset. Use obj.with_dataset(dataset)"
        )

    @contextmanager
    def with_dataset(self, dataset: AdjAndFeats):
        prev_dataset = self._dataset
        self._dataset = dataset
        try:
            yield
        finally:
            self._dataset = prev_dataset

    def preprocess_observation(self, obs: Observation) -> AdjAndFeats:
        graph_id, which_subgraphs, which_subgraphs_slices = observation_coos(obs)

        return self.dataset.at(
            graph_id,
            which_subgraphs,
            which_subgraphs_slices,
        ).to(next(self.parameters()).device)

    def compute_representations(
        self, batch: AdjAndFeats, t: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_repr = self.features_extractor(batch, t)  # (g, node, _)
        graph_repr = torch_scatter.segment_mean_csr(
            src=node_repr,
            indptr=batch.original_g_v_slices,
        )  # (g, _)
        node_repr = node_repr + graph_repr[batch.original_node2graph]

        node_unnorm_prob = self.node_selector(node_repr)  # (g, node, 1)

        if self.training:
            num_nodes = node_unnorm_prob.size(0)
            num2drop = int(self.drop_ratio * num_nodes)
            perm = torch.randperm(num_nodes)[:num2drop]
            node_unnorm_prob[perm] = -torch.inf

        return node_unnorm_prob

    def compute_probs(
        self,
        batch: AdjAndFeats,
        node_unnorm_prob: torch.Tensor,
        mask_value: torch.float,
    ) -> torch.Tensor:
        # Compute node probabilities
        unnormalized_probs = node_unnorm_prob.squeeze()

        dense_unnormalized_probs = torch.sparse_coo_tensor(
            torch.stack(
                (batch.original_node2graph, vrange(batch.num_original_nodes_per_graph))
            ),
            unnormalized_probs,
            requires_grad=True,
        ).to_dense()
        # TODO: remove above. don't create coo tensor, make a dense one right away and set the coos to the values
        #   as below
        #       TODO: double check whether below requires_grad and setting works. Otherwise
        #           do torch.scatter
        # dense_unnormalized_probs = torch.zeros(size=(batch.num_graphs, batch.num_original_nodes_per_graph.max()), device=batch.device, requires_grad=True)
        # dense_unnormalized_probs[batch.original_node2graph, vrange(batch.num_original_nodes_per_graph)] = unnormalized_probs

        mask = torch.full(
            size=(
                dense_unnormalized_probs.size(0),
                dense_unnormalized_probs.size(1)
                + 1,  # NOTE(first_node_id_is_1): 0 subgraph is the graph itself
            ),
            fill_value=dense_unnormalized_probs.max().item(),  # NOTE: sets fill value to max instead of 0. so min below masks the right thing
            device=batch.device,
        )
        # Mask nodes that have already been selected
        mask[batch.subgraph2graph, batch.subgraph_ids] = mask_value
        # Unmask graphs having less nodes than the number of subgraphs we want to select
        mask[batch.num_subgraphs_per_graph > batch.num_original_nodes_per_graph] = (
            dense_unnormalized_probs.max().item()
        )
        mask = mask[:, 1:]  # NOTE(first_node_id_is_1): 0 subgraph is the graph itself

        # Mask nodes that don't exist in the graphs
        ends = torch.full(
            size=(dense_unnormalized_probs.size(0),),
            fill_value=dense_unnormalized_probs.size(1),
            device=batch.device,
        )
        starts = batch.num_original_nodes_per_graph
        lengths = ends - starts
        cols = vrange(lengths, starts)
        rows = repeat_interleave(
            torch.arange(dense_unnormalized_probs.size(0), device=batch.device), lengths
        )
        mask[rows, cols] = mask_value

        dense_unnormalized_probs = torch.min(dense_unnormalized_probs, mask)

        return dense_unnormalized_probs

    def forward(
        self,
        batch: AdjAndFeats,
        t: int,
    ) -> Tuple[Tensor1D["batch_size"], Tensor2D["batch_size", "max_nodes"]]:
        if t == 0:
            self.probs = []
            self.indices = []
        node_unnorm_prob = self.compute_representations(batch, t)
        selection_probs = self.compute_probs(
            batch, node_unnorm_prob, mask_value=-torch.inf
        )

        if torch.isneginf(selection_probs.max(-1)[0]).any().item():
            raise Exception

        probs = selection_probs

        # Save for the visualization
        self.probs.append(probs.detach().cpu())

        # Sample using gumbel softmax
        samples, indices = top_k_gumbel_softmax(
            selection_probs,
            self.num_marked,
            self.tau,
            hard=True,
            use_noise=self.training,
        )

        self.indices.append(indices.unsqueeze(0))

        subgraphs = indices.flatten()

        return subgraphs, samples


class GumbelModel(torch.nn.Module):
    def __init__(
        self,
        selection_model: GumbelSelection,
        prediction_model: DSnetwork,
        num_subgraphs: int,
        num_hops: int = None,
    ):
        super().__init__()
        self.selection_model = selection_model
        self.prediction_model = prediction_model
        self.num_subgraphs = num_subgraphs
        self.num_hops = num_hops

    @contextmanager
    def with_dataset(self, dataset: AdjAndFeats):
        with self.selection_model.with_dataset(dataset):
            yield

    def mark_subgraphs(self, batched_data, samples, detach=False):
        # NOTE(first_node_id_is_1): for nodes of the subgraph of the original
        #  graph the marking is on no node and it does not carry gradient
        original = torch.zeros_like(samples[0])

        max_nodes = original.size(2)
        num_marked = original.size(-1)
        samples = torch.cat((original, *samples), dim=1).reshape(-1, num_marked)
        marking = samples[
            vrange(
                lengths=batched_data.num_nodes_per_subgraph,
                starts=torch.arange(
                    batched_data.num_total_subgraphs, device=samples.device
                )
                * max_nodes,
            )
        ]

        if debug.DEBUG:
            # Compute how many nodes can be marked to form a subgraph for each graph
            max_marked = torch.minimum(
                batched_data.num_original_nodes_per_graph, torch.tensor(num_marked)
            )
            # Repeat for the number of subgraphs
            max_marked = max_marked * (batched_data.num_subgraphs_per_graph - 1)

            # assert (marking.sum() == max_marked.sum()).item()

        if self.num_hops is not None:
            # Construct the ego networks
            batched_data = batched_data.construct_egonets(
                marking, num_hops=self.num_hops
            )
        if detach:
            marking = marking.detach()

        # NOTE: Use the marking to attach gradient
        node_repr = batched_data.v_features[:, num_marked:]
        return batched_data.replace(v_features=torch.cat((marking, node_repr), dim=-1))

    def forward(self, obs: Observation):
        samples = []
        for t in range(self.num_subgraphs - 1):
            batched_data = self.selection_model.preprocess_observation(obs)

            if len(samples) > 0:
                batched_data = self.mark_subgraphs(batched_data, samples)

            subgraphs, curr_samples = self.selection_model(batched_data, t)

            update_observation_inplace_no_replace(
                obs, subgraphs + 1
            )  # NOTE(first_node_id_is_1): 0 subgraph is the graph itself

            samples.append(curr_samples.unsqueeze(1))

        batched_data = self.selection_model.preprocess_observation(obs)
        batched_data = self.mark_subgraphs(batched_data, samples)
        out, _ = self.prediction_model(batched_data)
        return out


class RandomSelection(GumbelSelection):
    def __init__(
        self, num_marked, dataset: AdjAndFeats, num_subgraphs: int, num_hops: int = None
    ):
        super().__init__(num_marked, dataset, None, None, None, None, None, None)
        self._dataset = dataset
        self.num_subgraphs = num_subgraphs
        self.num_hops = num_hops

    def forward(
        self,
        obs: Observation,
    ) -> Observation:
        graphs = obs["graph_id"].squeeze()
        # return sample_subgraphs(
        #     self.dataset, graphs, self.num_subgraphs, self.num_subgraphs
        # )
        subgraphs_list = []
        sizes = []

        for i in graphs:
            num_subgraphs = self.dataset.num_subgraphs_per_graph[i].item()
            size = self.num_subgraphs - 1
            temp_subgraph = torch.hstack(
                (
                    torch.zeros(1, dtype=torch.int64),
                    torch.tensor(
                        np.random.choice(
                            num_subgraphs
                            - 1,  # NOTE(first_node_id_is_1): 0 subgraph is the graph itself
                            size=size,
                            replace=size > num_subgraphs - 1,
                        )
                        + 1
                    ),
                )
            )
            subgraphs_list.append(temp_subgraph)
            sizes.append(size + 1)  # Adding 1 to account for the zero added with hstack

        subgraphs = torch.hstack(subgraphs_list).to(self.dataset.device)

        subgraphs = torch.hstack(subgraphs_list).to(self.dataset.device)
        num_subgraphs_per_graph = torch.tensor(
            sizes, device=graphs.device, dtype=graphs.dtype
        )
        slices = torch.hstack(
            (
                num_subgraphs_per_graph.new_zeros((1,)),
                num_subgraphs_per_graph.cumsum(-1),
            )
        )
        batched_data = self.dataset.at(graphs, subgraphs, slices).to(graphs.device)

        return batched_data


class AllSelection(GumbelSelection):
    def __init__(
        self, num_marked, dataset: AdjAndFeats, num_subgraphs: int, num_hops: int = None
    ):
        super().__init__(num_marked, dataset, None, None, None, None, None, None)
        self._dataset = dataset
        self.num_subgraphs = num_subgraphs
        self.num_hops = num_hops

    def forward(
        self,
        obs: Observation,
    ) -> Observation:
        graphs = obs["graph_id"].squeeze()
        num_subgraphs_per_graph = self.dataset.num_subgraphs_per_graph[graphs]
        subgraphs = vrange(num_subgraphs_per_graph)
        slices = torch.hstack(
            (
                num_subgraphs_per_graph.new_zeros((1,)),
                num_subgraphs_per_graph.cumsum(-1),
            )
        )
        return self.dataset.at(graphs, subgraphs, slices).to(graphs.device)


class StandardModel(GumbelModel):
    def forward(self, obs: Observation):
        if self.num_subgraphs is not None:
            # obs = self.selection_model(obs)
            # batch = self.selection_model.preprocess_observation(obs)
            batch = self.selection_model(obs)
        else:
            batch = self.selection_model(obs)
        out, _ = self.prediction_model(batch.to(next(self.parameters()).device))
        return out
