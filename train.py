import copy
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, TypedDict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from torch.utils.data import DataLoader
from tqdm import trange

import config
import debug
import wandb
from data_utils import adj_and_feats_dataloader, dataset2tensor, mk_collate_fn
from datasets import get_data
from policy import (
    AllSelection,
    GumbelModel,
    GumbelSelection,
    RandomSelection,
    StandardModel,
)
from utils import setup

logger = logging.getLogger(__name__)


def get_prob_distr(
    batch, num_nodes, probs, visual, prefix, suffix, dirr, set_limit=True
):
    if len(probs) == 0 or visual is None or dirr is None:
        return {}

    ids = torch.isin(batch["graph_id"].cpu().squeeze(), visual)
    graph_ids = batch["graph_id"].cpu().squeeze()[ids]

    visual_values = {}
    for t, prob in enumerate(probs):
        curr_probs = prob[ids].detach().cpu()
        for gid, p in zip(graph_ids, curr_probs):
            plt.figure(figsize=(15, 10))
            if set_limit:
                plt.ylim(0.0, 1)
            plt.bar(
                x=np.arange(num_nodes[gid].item()), height=p[: num_nodes[gid]].numpy()
            )
            path = (dirr / str(uuid.uuid4())).with_suffix(".png")
            plt.savefig(path)
            visual_values[
                prefix + "/graph-" + str(gid.item()) + "/step-" + str(t) + "/" + suffix
            ] = wandb.Image(str(path))
            plt.close()
    return visual_values


def plot_selected_nodes_distribution(
    batch,
    num_nodes_per_graph,
    model,
    visual_eval: torch.Tensor = None,
    prefix: str = "test",
    dirr: Path = Path("/tmp"),
):
    model.eval()

    visual_probs = {}
    selected_nodes = []

    ids = torch.isin(batch["graph_id"].cpu().squeeze(), visual_eval)
    graph_ids = batch["graph_id"].cpu().squeeze()[ids]

    for i in range(10):
        _ = model(copy.deepcopy(batch))

        if i == 0:
            probs = model.selection_model.probs
            curr = get_prob_distr(
                batch,
                num_nodes_per_graph,
                probs,
                visual_eval,
                prefix=prefix,
                suffix="probs",
                dirr=dirr,
                set_limit=False,
            )
            visual_probs.update(curr)

        selected_nodes.append(torch.cat(model.selection_model.indices, dim=0))

    selected_nodes = torch.cat(selected_nodes, dim=-1).detach().cpu().numpy()
    for t in range(selected_nodes.shape[0]):
        curr_nodes = selected_nodes[t][ids]
        for gid, n in zip(graph_ids, curr_nodes):
            plt.figure(figsize=(15, 10))
            plt.ylim(0.0, 10)

            heights = np.zeros(num_nodes_per_graph[gid].item())
            values, counts = np.unique(n, return_counts=True)
            heights[values] = counts

            plt.bar(x=np.arange(num_nodes_per_graph[gid].item()), height=heights)

            path = (dirr / str(uuid.uuid4())).with_suffix(".png")
            plt.savefig(path)
            visual_probs[
                prefix
                + "/graph-"
                + str(gid.item())
                + "/step-"
                + str(t)
                + "/times_selected"
            ] = wandb.Image(str(path))
            plt.close()

    return visual_probs


def eval(
    loader,
    model,
    metric_fn,
    std: torch.Tensor = None,
):
    model.eval()

    all_y_pred = []
    for i in range(1):
        y_true, y_pred = [], []
        for batch in loader:
            pred = model(batch)

            y_true.append(batch["ys"].cpu())
            y_pred.append(pred.detach().cpu())

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        all_y_pred.append(y_pred.unsqueeze(-1))

    y_pred = torch.cat(all_y_pred, dim=-1).mean(-1)

    metric = (
        metric_fn(y_pred, y_true) if std is None else metric_fn(y_pred, y_true, std)
    )
    return metric


def train_one_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    selection_optimizer: torch.optim.Optimizer,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], float],
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
) -> float:
    model.train()

    num_skipped = 0
    total_loss = 0
    num_elements_trained_on = 0
    for batch in loader:
        is_labeled = batch["ys"] == batch["ys"]

        try:
            if selection_optimizer is not None:
                selection_optimizer.zero_grad()
            optimizer.zero_grad()

            pred = model(batch)

            loss = loss_fn(
                pred[is_labeled], batch["ys"].float()[is_labeled].to(pred.device)
            )
            loss.backward()
            if selection_optimizer is not None:
                selection_optimizer.step()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        except Exception:
            num_skipped += batch["graph_id"].size(0)
            continue

        num_elements_trained_on += batch["graph_id"].size(0)
        total_loss += loss.item() * batch["graph_id"].size(0)

    return (total_loss / (num_elements_trained_on + 1e-10), num_skipped)


class Checkpoint(TypedDict):
    epoch: int
    model: Dict
    selection_optimizer: Dict
    optimizer: Dict
    scheduler: Any
    val_metric: float


def train(
    model,
    selection_optimizer,
    optimizer,
    scheduler,
    loss_fn,
    metric_fn,
    should_checkpoint: config.ShouldCheckpoint,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    ckpt_path: Path = Path("ckpts/ckpt.pt"),
    std: torch.Tensor = None,
):
    def info(log):
        logger.info(f"{train.__name__}: {log}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    (ckpt_path.parent / "imgs").mkdir()

    best_epoch, best_metric = 0, should_checkpoint.init
    for epoch in trange(num_epochs, desc="Training..", mininterval=5, leave=False):
        with model.with_dataset(train_loader.collate_fn.dataset):
            train_loss, num_skipped = train_one_epoch(
                train_loader,
                model,
                selection_optimizer,
                optimizer,
                loss_fn,
                (
                    scheduler
                    if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
                    else None
                ),
            )

        with model.with_dataset(val_loader.collate_fn.dataset):
            val_metric = eval(
                val_loader,
                model,
                metric_fn,
                std=std,
            )

        if should_checkpoint.is_better(val_metric, best_metric):
            torch.save(
                Checkpoint(
                    epoch=epoch,
                    model=model.state_dict(),
                    selection_optimizer=(
                        selection_optimizer.state_dict()
                        if selection_optimizer is not None
                        else {}
                    ),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    val_metric=val_metric,
                ),
                ckpt_path,
            )
            best_epoch, best_metric = epoch, val_metric

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metric)
            lr = optimizer.param_groups[0]["lr"]
        else:
            if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                scheduler.step()
            lr = scheduler.get_last_lr()[0]

        visual_test_value = {}
        if debug.DEBUG and epoch % 20 == 0 and selection_optimizer is not None:
            with model.with_dataset(test_loader.collate_fn.dataset):
                test_metric = eval(test_loader, model, metric_fn, std=std)
                batch = next(iter(test_loader))
                num_nodes_per_graph = (
                    test_loader.collate_fn.dataset.num_original_nodes_per_graph
                )
                visual_test_value = plot_selected_nodes_distribution(
                    batch,
                    num_nodes_per_graph,
                    model,
                    visual_eval=torch.arange(3),
                    dirr=ckpt_path.parent / "imgs",
                )

        wandb.log(
            {
                "train_loss": train_loss,
                "val_metric": val_metric,
                "lr": lr,
                "selection_lr": (
                    selection_optimizer.param_groups[0]["lr"]
                    if selection_optimizer is not None
                    else 0
                ),
                "num_skipped": num_skipped,
            }
        )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if optimizer.param_groups[0]["lr"] < 0.00001:
                break

    info(f"Finished, {epoch=} {best_epoch=} {best_metric=:.4f} '{str(ckpt_path)}'")

    with model.with_dataset(test_loader.collate_fn.dataset):
        test_metric = eval(test_loader, model, metric_fn, std=std)

        if debug.DEBUG and selection_optimizer is not None:
            batch = next(iter(test_loader))
            num_nodes_per_graph = (
                test_loader.collate_fn.dataset.num_original_nodes_per_graph
            )
            visual_test_value = plot_selected_nodes_distribution(
                batch,
                num_nodes_per_graph,
                model,
                visual_eval=torch.arange(3),
                dirr=ckpt_path.parent / "imgs",
            )
            wandb.log(visual_test_value)

    logger.info(f"Trained model -- Test metric: {test_metric:.4f}")
    wandb.run.summary["test metric"] = test_metric

    info(f"Loading checkpoint '{str(ckpt_path)}'.")
    checkpoint: Checkpoint = torch.load(ckpt_path)
    info(
        f"Checkpoint info: epoch={checkpoint['epoch']}, val_metric={checkpoint['val_metric']:.4f}"
    )
    model.load_state_dict(checkpoint["model"])

    with model.with_dataset(test_loader.collate_fn.dataset):
        test_metric = eval(test_loader, model, metric_fn, std=std)
    logger.info(f"Trained model -- Test metric @ best val: {test_metric:.4f}")
    wandb.run.summary["best val"] = checkpoint["val_metric"]
    wandb.run.summary["test metric @ best val"] = test_metric


@hydra.main(config_path="conf", config_name="train", version_base="1.3")
def main(cfg: config.TrainConfig):
    run_directory = setup(cfg)

    ckpt_path = run_directory / str(time.time_ns()) / "chkpt.pt"
    with omegaconf.open_dict(cfg):
        cfg.ckpt_path = ckpt_path

    wandb.init(project="gumbel", config=omegaconf.OmegaConf.to_container(cfg))

    train_dataset, val_dataset, test_dataset = get_data(
        cfg.task.dataset_name, cfg.task.dataroot, cfg.split
    )
    std = train_dataset.std if hasattr(train_dataset, "std") else None

    train_dataset = dataset2tensor(train_dataset, num_marked=cfg.num_marked).to(
        cfg.data_device
    )
    val_dataset = dataset2tensor(val_dataset, num_marked=cfg.num_marked).to(
        cfg.data_device
    )
    test_dataset = dataset2tensor(test_dataset, num_marked=cfg.num_marked).to(
        cfg.data_device
    )

    train_loader = adj_and_feats_dataloader(
        train_dataset,
        cfg.batch_size,
        collate_fn=mk_collate_fn(
            train_dataset,
            cfg.num_subgraphs,
        ),
        shuffle=True,
    )
    val_loader = adj_and_feats_dataloader(
        val_dataset,
        cfg.batch_size,
        collate_fn=mk_collate_fn(
            val_dataset,
            cfg.num_subgraphs,
        ),
        shuffle=False,
    )
    test_loader = adj_and_feats_dataloader(
        test_dataset,
        cfg.batch_size,
        collate_fn=mk_collate_fn(
            test_dataset,
            cfg.num_subgraphs,
        ),
        shuffle=False,
    )

    model_backbone = hydra.utils.instantiate(cfg.model_backbone)
    prediction_model = hydra.utils.instantiate(
        cfg.model, backbone=model_backbone, return_x=True
    ).to(cfg.device)

    if cfg.selection_type == "gumbel":
        model_backbone = hydra.utils.instantiate(
            cfg.selection_backbone, num_subgraphs=cfg.num_subgraphs
        )
        _selection_model = hydra.utils.instantiate(
            cfg.selection_model, backbone=model_backbone
        )
        selection_model = GumbelSelection(
            num_marked=cfg.num_marked,
            num_conv_steps=len(prediction_model.backbone.layers),
            dataset=train_dataset,
            features_extractor=_selection_model,
            tau=cfg.gumbel_softmax_tau,
            num_subgraphs=cfg.num_subgraphs,
            drop_ratio=cfg.drop_ratio,
            downstream_emb_dim=prediction_model.backbone.emb_dim,
        ).to(cfg.device)

        model = GumbelModel(
            selection_model,
            prediction_model,
            cfg.num_subgraphs,
            cfg.task.num_hops,
        ).to(cfg.device)

        selection_optimizer = hydra.utils.instantiate(
            cfg.selection_opt.optimizer, params=selection_model.parameters(), eps=1e-3
        )
    elif cfg.selection_type == "random" or cfg.selection_type == "all":
        if cfg.selection_type == "all":
            assert cfg.num_subgraphs is None
        selection_class = (
            RandomSelection if cfg.selection_type == "random" else AllSelection
        )
        selection_model = selection_class(
            num_marked=cfg.num_marked,
            dataset=train_dataset,
            num_subgraphs=cfg.num_subgraphs,
            num_hops=cfg.task.num_hops,
        ).to(cfg.device)

        model = StandardModel(
            selection_model,
            prediction_model,
            cfg.num_subgraphs,
        ).to(cfg.device)

        selection_optimizer = None
    else:
        NotImplementedError(f"{cfg.selection_type=}")

    optimizer = hydra.utils.instantiate(
        cfg.opt.optimizer, params=prediction_model.parameters(), eps=1e-3
    )

    sched_kwargs = dict(optimizer=optimizer)
    if issubclass(
        omegaconf.OmegaConf.get_type(cfg.opt.scheduler), config.LambdaLRConfig
    ):
        if cfg.opt.scheduler.num_training_steps is None:
            sched_kwargs["num_training_steps"] = cfg.opt.num_epochs * len(train_loader)
        sched_kwargs["num_warmup_steps"] = cfg.opt.scheduler.num_warmup_epochs * len(
            train_loader
        )
        with omegaconf.open_dict(cfg.opt.scheduler):
            del cfg.opt.scheduler["num_warmup_epochs"]
    scheduler = hydra.utils.instantiate(cfg.opt.scheduler, **sched_kwargs)
    loss_fn = hydra.utils.call(cfg.task.loss)
    metric_fn = hydra.utils.call(cfg.task.metric)

    train(
        model,
        selection_optimizer,
        optimizer,
        scheduler,
        loss_fn,
        metric_fn,
        should_checkpoint=cfg.should_checkpoint,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=cfg.opt.num_epochs,
        ckpt_path=ckpt_path,
        std=std,
    )

    wandb.join()


if __name__ == "__main__":
    # import os
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    main()
