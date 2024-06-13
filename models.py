from typing import Callable, List

import torch
import torch.nn.functional as F
import torch_scatter
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.nn import GINConv, GINEConv

from data_utils import AdjAndFeats


class AlchemyCustomGINE(torch.nn.Module):
    def __init__(self, in_dim, emb_dim, layernorm, track_running_stats, num_edge_emb=4):
        super().__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.layer = GINEConv(nn=mlp, train_eps=True)
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.Linear(num_edge_emb, in_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_dim, in_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, self.edge_embedding(edge_attr))

    def reset_parameters(self):
        self.edge_embedding.reset_parameters()
        self.layer.reset_parameters()


class CustomGINE(torch.nn.Module):
    def __init__(self, in_dim, emb_dim, layernorm, track_running_stats, num_edge_emb=4):
        super().__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            (
                torch.nn.BatchNorm1d(emb_dim, track_running_stats=track_running_stats)
                if not layernorm
                else torch.nn.LayerNorm(emb_dim)
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.layer = GINEConv(nn=mlp, train_eps=True)
        self.edge_embedding = torch.nn.Embedding(
            num_embeddings=num_edge_emb, embedding_dim=in_dim
        )

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, self.edge_embedding(edge_attr))

    def reset_parameters(self):
        self.edge_embedding.reset_parameters()
        self.layer.reset_parameters()


class OGBGCustomGINE(torch.nn.Module):
    def __init__(self, in_dim, emb_dim, layernorm, track_running_stats):
        super().__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            (
                torch.nn.BatchNorm1d(emb_dim, track_running_stats=track_running_stats)
                if not layernorm
                else torch.nn.LayerNorm(emb_dim)
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.layer = GINEConv(nn=mlp, train_eps=True)
        self.edge_embedding = BondEncoder(emb_dim=in_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, self.edge_embedding(edge_attr))

    def reset_parameters(self):
        self.edge_embedding.reset_parameters()
        self.layer.reset_parameters()


class CustomGIN(torch.nn.Module):
    def __init__(self, in_dim, emb_dim, layernorm, track_running_stats, num_edge_emb=4):
        super().__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            (
                torch.nn.BatchNorm1d(emb_dim, track_running_stats=track_running_stats)
                if not layernorm
                else torch.nn.LayerNorm(emb_dim)
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.layer = GINConv(nn=mlp, train_eps=True)

    def forward(self, x, edge_index, edge_attr=None):
        return self.layer(x, edge_index)

    def reset_parameters(self):
        self.layer.reset_parameters()


class DSLayer(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        emb_dim,
        pos_in_dim,
        GNNConv: Callable[[int, int], torch.nn.Module],
        layernorm: bool,
        add_residual: bool,
        track_running_stats: bool,
        model_drop_ratio: float,
    ):
        super().__init__()

        self.gnn = GNNConv(
            in_dim + pos_in_dim,
            emb_dim,
            layernorm,
            track_running_stats=track_running_stats,
        )
        self.bn = (
            torch.nn.BatchNorm1d(emb_dim, track_running_stats=track_running_stats)
            if not layernorm
            else torch.nn.LayerNorm(emb_dim)
        )

        self.gnn_pos = CustomGIN(
            pos_in_dim, emb_dim, layernorm, track_running_stats=track_running_stats
        )
        self.bn_pos = (
            torch.nn.BatchNorm1d(emb_dim, track_running_stats=track_running_stats)
            if not layernorm
            else torch.nn.LayerNorm(emb_dim)
        )

        self.add_residual = add_residual
        self.model_drop_ratio = model_drop_ratio

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.bn.reset_parameters()
        self.gnn_sum.reset_parameters()
        self.bn_sum.reset_parameters()

    def forward(self, batched_data: AdjAndFeats, pos_embeddings) -> AdjAndFeats:
        x, edge_index, edge_attr = (
            batched_data.v_features,
            batched_data.flat_edge_index,
            batched_data.e_features,
        )
        node_feats = torch.cat((x, pos_embeddings), dim=-1)

        h = torch.relu(
            self.bn(self.gnn(node_feats, edge_index, edge_attr))
        )  # (g, subg, node, _)
        p = torch.relu(
            self.bn_pos(self.gnn_pos(pos_embeddings, edge_index, edge_attr))
        )  # (g, subg, node, _)

        h = F.dropout(h, self.model_drop_ratio, training=self.training)

        if self.add_residual:
            h = h + x
        return batched_data.replace(v_features=h), p


class DSnetworkBackbone(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        emb_dim,
        feature_encoder,
        GNNConv,
        layernorm=False,
        add_residual=False,
        track_running_stats=True,
        num_subgraphs=None,
        model_drop_ratio=0.0,
    ):
        super().__init__()

        self.emb_dim = emb_dim

        self.feature_encoder = feature_encoder

        self.init_linear = None
        if add_residual and in_dim != emb_dim:
            self.init_linear = torch.nn.Linear(in_dim, emb_dim)
            in_dim = emb_dim

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                DSLayer(
                    emb_dim if i != 0 else in_dim,
                    emb_dim,
                    emb_dim if i != 0 else self.feature_encoder.k,
                    GNNConv,
                    layernorm,
                    add_residual,
                    track_running_stats,
                    model_drop_ratio,
                )
            )

    def reset_parameters(self):
        def fn(submodule):
            reset_feat = getattr(submodule, "reset_parameters", None)
            if callable(reset_feat):
                reset_feat()

        fn(self.feature_encoder)

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, batched_data: AdjAndFeats, t: int):
        x = batched_data.v_features

        x = self.feature_encoder(x)  # (g, subg, node, _)
        pos_embedding = x[:, : self.feature_encoder.k]

        if self.init_linear is not None:
            x = self.init_linear(x)

        batched_data = batched_data.replace(v_features=x)

        for layer in self.layers:
            batched_data, pos_embedding = layer(batched_data, pos_embedding)

        return batched_data.replace(v_features=batched_data.v_features + pos_embedding)


class GNNnetworkBackbone(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        emb_dim,
        feature_encoder,
        GNNConv,
        layernorm=False,
        add_residual=False,
        track_running_stats=True,
    ):
        super().__init__()

        self.emb_dim = emb_dim

        self.feature_encoder = feature_encoder

        self.gnn = GNNnetwork(
            num_layers,
            in_dim=self.feature_encoder.k,
            emb_dim=emb_dim,
            feature_encoder=feature_encoder,
            GNNConv=GNNConv,
            layernorm=layernorm,
            add_residual=add_residual,
            track_running_stats=track_running_stats,
        )

    def forward(self, batched_data: AdjAndFeats):
        x, edge_index = (
            batched_data.v_features,
            batched_data.flat_edge_index,
        )

        labels = self.gnn(x, edge_index)

        return batched_data.replace(v_features=labels)


class DSnetwork(torch.nn.Module):
    def __init__(
        self,
        backbone: DSnetworkBackbone,
        final_reductions: List[str],
        num_tasks: int = None,
        return_x: bool = False,
    ):
        super().__init__()

        self.backbone = backbone
        self.final_reductions = final_reductions

        self.final_layers = None
        if num_tasks is not None:
            emb_dim = backbone.emb_dim
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 * emb_dim, out_features=num_tasks),
            )
        self.return_x = return_x

    def reset_parameters(self):
        def fn(submodule):
            reset_feat = getattr(submodule, "reset_parameters", None)
            if callable(reset_feat):
                reset_feat()

        self.backbone.reset_parameters()

        if self.final_layers is not None:
            self.final_layers.apply(fn=fn)

    def forward(self, batched_data: AdjAndFeats, t: int = None):
        batched_data = self.backbone(batched_data, t=t)
        return self.reduce_and_predict(batched_data)

    def reduce_and_predict(self, batched_data):
        x = batched_data.v_features

        final_reductions = self.final_reductions
        if final_reductions[0][0] == "node":
            x = torch_scatter.segment_csr(
                src=x, indptr=batched_data.v_subg_slices, reduce=final_reductions[0][1]
            )  # : (g, subg, node, _) -> (g, subg, _)

            if len(final_reductions) > 1:
                assert (
                    len(final_reductions) == 2 and final_reductions[1][0] == "subg"
                ), f"{final_reductions=}"
                x = torch_scatter.segment_csr(
                    src=x,
                    indptr=batched_data.g_slices,
                    reduce=final_reductions[1][1],
                )  # : (g, subg, _) -> (g, _)

        elif final_reductions[0][0] == "subg":
            x = torch_scatter.scatter(
                src=x,
                index=batched_data.global_node_id,
                dim=0,
                reduce=final_reductions[0][1],
            )  # : (g, subg, node, _) -> (g, node, _)
            if len(final_reductions) > 1:
                assert (
                    len(final_reductions) == 2 and final_reductions[1][0] == "node"
                ), f"{final_reductions=}"
                x = torch_scatter.segment_csr(
                    src=x,
                    indptr=batched_data.original_g_v_slices,
                    reduce=final_reductions[1][1],
                )  # : (g, node, _) -> (g, _)
        else:
            NotImplementedError(f"{final_reductions=}")

        # # Needed by downstream graph classification task
        # [
        #     ("node", mean),
        #     ("subg", mean),
        # ]

        # [
        #     ("subg", mean),
        #     ("node", mean)
        # ]

        # # Needed by policy to select node
        # [
        #     ("subg", mean)
        # ]

        # # Also maybe
        # [
        #     ("node", mean)
        # ]

        out = x
        if self.final_layers is not None:
            out = self.final_layers(x)
        return (out, x) if self.return_x else out


class GNNnetwork(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        emb_dim,
        feature_encoder,
        GNNConv,
        layernorm=False,
        add_residual=False,
        track_running_stats=True,
    ):
        super().__init__()

        self.emb_dim = emb_dim

        self.feature_encoder = feature_encoder

        self.gnn_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                GNNConv(
                    emb_dim if i != 0 else in_dim,
                    emb_dim,
                    layernorm,
                    track_running_stats=track_running_stats,
                )
            )
            self.bn_layers.append(
                torch.nn.BatchNorm1d(emb_dim, track_running_stats=track_running_stats)
                if not layernorm
                else torch.nn.LayerNorm(emb_dim)
            )

    def reset_parameters(self):
        def fn(submodule):
            reset_feat = getattr(submodule, "reset_parameters", None)
            if callable(reset_feat):
                reset_feat()

        fn(self.feature_encoder)

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, batched_data: AdjAndFeats):
        x, edge_index, edge_attr = (
            batched_data.v_features,
            batched_data.flat_edge_index,
            batched_data.e_features,
        )
        x = self.feature_encoder(x)  # (g, node, _)

        for gnn, bn in zip(self.gnn_layers, self.bn_layers):
            x = torch.relu(bn(gnn(x, edge_index, edge_attr)))

        return x
