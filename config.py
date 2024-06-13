import copy
import importlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional

import omegaconf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

###############################################
# SECTION: AUXILIARY DEFINITIONS
###############################################


@dataclass
class SafeCallable:
    callable: Any
    _target_: str = f"{__name__}.SafeCallable"

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.callable(*args, **kwds)

    def __str__(self):
        return str(self.callable.__name__)


def fqn_import(fqn: str, *args):
    module_name, _, function_name = fqn.rpartition(".")
    fn = getattr(importlib.import_module(module_name), function_name)
    if len(args) != 0:
        fn = partial(fn, *args)
    return SafeCallable(fn)


CALLABLE_RESOLVER = "callable"
OmegaConf.register_new_resolver(CALLABLE_RESOLVER, fqn_import)


###############################################
# SECTION: OPTIMIZERS
###############################################


@dataclass
class OptimizerConfig:
    _target_: str
    lr: float = 0.001


@dataclass
class AdamConfig(OptimizerConfig):
    _target_: str = "torch.optim.Adam"


@dataclass
class SGDConfig(OptimizerConfig):
    _target_: str = "torch.optim.SGD"


@dataclass
class RMSpropConfig(OptimizerConfig):
    _target_: str = "torch.optim.RMSprop"


###############################################
# SECTION: SCHEDULERS
###############################################


@dataclass
class SchedulerConfig:
    _target_: str


@dataclass
class StepLRConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    step_size: int = 50
    gamma: float = 0.5


@dataclass
class MultiStepLRConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.MultiStepLR"
    milestones: Any = MISSING
    gamma: float = 0.5


@dataclass
class ConstantLRConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.ConstantLR"
    factor: int = 1


@dataclass
class ReduceLROnPlateauConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    mode: str = "min"
    factor: float = 0.5
    patience: int = 40


@dataclass
class LambdaLRConfig(SchedulerConfig):
    _target_: str = "data_utils.get_linear_schedule_with_warmup"
    num_warmup_epochs: int = 10
    num_training_steps: Optional[int] = None
    last_epoch: int = -1


###############################################
# SECTION: SCHED AND OPTIM BASE CLASS
###############################################


@dataclass
class OptimConfig:
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    num_epochs: int


###############################################
# SECTION: FEATURE ENCODERS
###############################################


@dataclass
class FeatureEncoderConfig:
    _target_: str


@dataclass
class Identity(FeatureEncoderConfig):
    k: int = MISSING
    in_dim: int = MISSING
    emb_dim: int = MISSING
    _target_: str = "feature_encoders.Identity"


@dataclass
class ZincAtomEncoder(FeatureEncoderConfig):
    k: int = MISSING
    in_dim: int = MISSING
    emb_dim: int = MISSING
    _target_: str = "feature_encoders.ZincAtomEncoder"


@dataclass
class MyAtomEncoder(FeatureEncoderConfig):
    k: int = MISSING
    in_dim: int = MISSING
    emb_dim: int = MISSING
    _target_: str = "feature_encoders.MyAtomEncoder"


@dataclass
class LinearEncoder(FeatureEncoderConfig):
    k: int = MISSING
    in_dim: int = MISSING
    emb_dim: int = MISSING
    _target_: str = "feature_encoders.LinearEncoder"


@dataclass


###############################################
# SECTION: DS
###############################################


@dataclass
class DSNetworkArgs:
    _target_: str = "builtins.dict"
    # Task defined options
    num_tasks: Optional[int] = MISSING
    final_reductions: Any = MISSING
    return_x: bool = False


@dataclass
class DSNetworkBackbone:
    _target_: str = "models.DSnetworkBackbone"
    # Task defined options
    in_dim: int = MISSING
    feature_encoder: Any = MISSING

    # Decent defaults
    num_layers: int = 6
    emb_dim: int = 128
    GNNConv: SafeCallable = f"${{{CALLABLE_RESOLVER}:torch_geometric.nn.GraphConv}}"
    layernorm: bool = False
    add_residual: bool = False
    track_running_stats: bool = True
    model_drop_ratio: float = 0.0


@dataclass
class GNNNetworkBackbone:
    _target_: str = "models.GNNnetworkBackbone"
    # Task defined options
    in_dim: int = MISSING
    feature_encoder: Any = MISSING

    # Decent defaults
    num_layers: int = 6
    emb_dim: int = 128
    GNNConv: SafeCallable = f"${{{CALLABLE_RESOLVER}:torch_geometric.nn.GraphConv}}"
    layernorm: bool = False
    add_residual: bool = False
    track_running_stats: bool = True


@dataclass
class DSNetworkConfig(DSNetworkArgs):
    _target_: str = "models.DSnetwork"


@dataclass
class GNNNetworkConfig:
    _target_: str = "models.GNNnetwork"
    # Task defined options
    in_dim: int = MISSING
    feature_encoder: Any = MISSING

    # Decent defaults
    num_layers: int = 6
    emb_dim: int = 128
    GNNConv: SafeCallable = f"${{{CALLABLE_RESOLVER}:torch_geometric.nn.GraphConv}}"
    layernorm: bool = False
    add_residual: bool = False
    track_running_stats: bool = True


@dataclass
class TaskConfig:
    dataroot: Path
    dataset_name: str
    num_tasks: int
    num_hops: Optional[int]
    in_dim: int
    model_feature_encoder: FeatureEncoderConfig
    selection_feature_encoder: FeatureEncoderConfig
    metric: SafeCallable
    loss: SafeCallable


@dataclass
class ShouldCheckpoint:
    init: float
    is_better: SafeCallable


@dataclass
class TrainConfig:
    task: TaskConfig
    model_backbone: DSNetworkBackbone
    model: DSNetworkConfig
    selection_backbone: DSNetworkBackbone
    selection_model: DSNetworkConfig
    opt: OptimConfig
    selection_opt: OptimConfig
    should_checkpoint: ShouldCheckpoint
    batch_size: int
    selection_type: str
    num_marked: int
    num_subgraphs: Optional[int]
    gumbel_softmax_tau: float = 2.0 / 3.0
    drop_ratio: float = 0.0
    seed: int = 1337
    device: str = "cuda"
    data_device: str = "cuda"
    split: int = 0


cs = ConfigStore.instance()
for node in [
    SGDConfig,
    RMSpropConfig,
    AdamConfig,
    StepLRConfig,
    MultiStepLRConfig,
    ConstantLRConfig,
    ReduceLROnPlateauConfig,
    LambdaLRConfig,
]:
    cs.store(group="optim", name=node.__name__, node=node)

for node in [
    DSNetworkArgs,
    DSNetworkConfig,
    DSNetworkBackbone,
    GNNNetworkConfig,
    GNNNetworkBackbone,
]:
    cs.store(group="models", name=node.__name__, node=node)

for node in [Identity, ZincAtomEncoder, MyAtomEncoder, LinearEncoder]:
    cs.store(group="feature_encoders", name=node.__name__, node=node)

cs.store(group="task", name="TaskConfig", node=TaskConfig)
cs.store(name="TrainConfig", node=TrainConfig)


def resolve(config: omegaconf.dictconfig.DictConfig) -> omegaconf.dictconfig.DictConfig:
    config_copy = copy.deepcopy(config)
    config_copy._set_flag(
        flags=["allow_objects", "struct", "readonly"], values=[True, False, False]
    )
    config_copy._set_parent(config._get_parent())
    OmegaConf.resolve(config_copy)
    return config_copy
