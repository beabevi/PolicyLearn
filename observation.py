from typing import Generic, Tuple, TypedDict, TypeVar

import numpy as np
import torch
from torch.profiler import record_function

import debug

T1 = TypeVar("T1")
T2 = TypeVar("T2")


class Tensor1D(torch.Tensor, Generic[T1]): ...


class Tensor2D(torch.Tensor, Generic[T1, T2]): ...


# Observation is in practice a sparse matrix (or ragged tensor)
# containing for each graph a few subgraph ids (varying number of
# subgraphs across graphs) padded to a dense matrix with fill value -1.
# The padding is necessary because SB3's rollout buffer requires
# static shapes across observations.
class Observation(TypedDict):
    graph_id: Tensor2D["task_bsz", 1]
    which_subgraphs: Tensor2D["task_bsz", "max_subgraphs"]
    ys: Tensor2D["task_bsz", "y_d"]


def observation_to(obs: Observation, device: torch.device):
    return Observation(
        graph_id=obs["graph_id"].to(device),
        which_subgraphs=obs["which_subgraphs"].to(device),
        ys=obs["ys"].to(device),
    )


@record_function("observation_coos")
def observation_coos(
    obs: Observation,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns the coordinates of an observation (graph_ids, subgraph_ids)
    """
    graph_id, which_subgraphs = (
        obs["graph_id"],
        obs["which_subgraphs"],
    )
    nnmt = which_subgraphs != -1

    nneg_rows_cols = nnmt.nonzero()
    at1 = which_subgraphs[nneg_rows_cols[:, 0], nneg_rows_cols[:, 1]].squeeze()

    num_subgraphs_per_graph = nnmt.sum(-1)
    return (
        graph_id.squeeze(),
        at1,
        torch.hstack(
            (
                num_subgraphs_per_graph.new_zeros((1,)),
                num_subgraphs_per_graph.cumsum(-1),
            )
        ),
    )


def update_observation_inplace(
    obs: Observation, action: Tensor1D["self.task_bsz"], max_num_nodes
):
    which_subgraphs = obs["which_subgraphs"]
    if isinstance(action, np.ndarray):
        action = torch.tensor(action, device=which_subgraphs.device)
    elif isinstance(action, torch.Tensor):
        action = action.to(which_subgraphs.device)

    remove_subg = (
        action // max_num_nodes + 1
    )  # NOTE(first_node_id_is_1): 0 subgraph is the graph itself
    add_subg = (
        action % max_num_nodes + 1
    )  # NOTE(first_node_id_is_1): 0 subgraph is the graph itself

    if debug.DEBUG:
        # Ensure we remove one subgraph in the range
        assert remove_subg.max().item() < which_subgraphs.size(-1)
        assert (
            which_subgraphs[
                torch.arange(which_subgraphs.size(0), device=which_subgraphs.device),
                remove_subg,
            ]
            != -1
        ).all()

    which_subgraphs[
        torch.arange(which_subgraphs.size(0), device=which_subgraphs.device),
        remove_subg,
    ] = add_subg  # NOTE(first_node_id_is_1): 0 subgraph is the graph itself
    return obs


def update_observation_inplace_no_replace(
    obs: Observation, action: Tensor1D["self.task_bsz"]
):
    which_subgraphs = obs["which_subgraphs"]
    if isinstance(action, np.ndarray):
        action = torch.tensor(action, device=which_subgraphs.device)
    elif isinstance(action, torch.Tensor):
        action = action.to(which_subgraphs.device)
    (nnz,) = action.nonzero(as_tuple=True)
    if debug.DEBUG:
        # Ensure there are spots available
        assert (which_subgraphs == -1).any(-1).all().item()
    add_position = which_subgraphs[nnz].argmin(-1)

    which_subgraphs[nnz, add_position] = action[
        nnz
    ]  # NOTE(first_node_id_is_1): 0 subgraph is the graph itself
    return obs
