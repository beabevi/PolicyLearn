import logging
import random
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import omegaconf
import torch
import torchmetrics
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from sklearn.metrics import roc_auc_score


logger = logging.getLogger(__name__)


def git_logs(run_directory: Path):
    try:
        script_directory = Path(__file__).resolve().parent
        dirty = subprocess.call(shlex.split("git diff-index --quiet HEAD --"))
        if dirty != 0:
            logger.warning(".git dirty")
            with open(run_directory / "dirty.diff", "w") as f:
                err = subprocess.call(shlex.split("git diff"), stdout=f, stderr=f)
                if err != 0:
                    logger.warning("Couldn't create diff")
        git_hash = (
            subprocess.check_output(
                shlex.split("git describe --always"), cwd=script_directory
            )
            .strip()
            .decode()
        )
        logger.info(f"Git hash: {git_hash}")
    except subprocess.CalledProcessError:
        logger.warning("Could not retrieve git hash")


def setup(cfg):
    hydra_conf = HydraConfig.get()
    missing = omegaconf.OmegaConf.missing_keys(cfg)
    if len(missing) > 0:
        logger.fatal(f"Missing keys in configuration: {', '.join(missing)}")
        exit(1)

    run_directory = Path(hydra_conf.run.dir)
    if hydra_conf.mode == RunMode.MULTIRUN:
        run_directory = Path(hydra_conf.sweep.dir) / hydra_conf.sweep.subdir

    logger.info(f"Run directory {run_directory}")
    git_logs(run_directory)

    logger.info(f"{shlex.join(['python', *sys.argv])}")

    logger.info(f"Your lucky number is {cfg.seed}")
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    # torch.use_deterministic_algorithms(True)

    return run_directory


def top_k_gumbel_softmax(
    logits: torch.Tensor,
    k: int,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
    use_noise=True,
) -> torch.Tensor:
    # FIXME(beabevi): Consider removing this dependency on the max number of nodes
    if use_noise:
        gumbels = torch.distributions.Gumbel(
            torch.zeros_like(logits), torch.ones_like(logits)
        ).sample()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    else:
        gumbels = logits
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        indices = y_soft.topk(k, dim)[1]
        rets = []
        for i in range(k):
            y_hard = torch.zeros_like(logits).scatter_(dim, indices[:, i : i + 1], 1.0)
            ret = y_hard - y_soft.detach() + y_soft
            rets.append(ret.unsqueeze(-1))
        ret = torch.cat(rets, dim=-1)
    else:
        # Reparametrization trick.
        indices = None
        ret = y_soft
    return ret, indices


def root_mean_squared_error(preds, target):
    return torchmetrics.functional.mean_squared_error(preds, target, squared=False)


def rocauc(preds, targets):
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    rocauc_list = []

    for i in range(targets.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.any(targets[:, i] == 1) and np.any(targets[:, i] == 0):
            # ignore nan values
            is_labeled = targets[:, i] == targets[:, i]
            rocauc_list.append(
                roc_auc_score(targets[is_labeled, i], preds[is_labeled, i])
            )

    if len(rocauc_list) == 0:
        raise RuntimeError(
            "No positively labeled data available. Cannot compute ROC-AUC."
        )

    return sum(rocauc_list) / len(rocauc_list)


def mean_absolute_error_with_std(preds, target, std):
    return torchmetrics.functional.mean_absolute_error(preds * std, target * std)
