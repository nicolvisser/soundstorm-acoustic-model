from dataclasses import dataclass
from typing import List, Optional, Tuple

from simple_parsing import Serializable


@dataclass
class SoundStormTrainerArgs(Serializable):
    # run name
    run_name: str
    # dataloader
    conditioning_tokens_train_dataset_paths: List[str]
    conditioning_tokens_val_dataset_paths: List[str]
    codec_tokens_train_dataset_paths: List[str]
    codec_tokens_val_dataset_paths: List[str]
    batch_size: int
    num_workers: int
    # optimizer
    lr_max: float
    betas: Tuple[float, float]
    weight_decay: float
    warmup_steps: int
    decay_steps: int
    lr_final: float
    # pl.trainer
    accelerator: str
    fast_dev_run: bool
    max_epochs: Optional[int]
    min_epochs: Optional[int]
    max_steps: int
    min_steps: Optional[int]
    val_check_interval: Optional[float]
    check_val_every_n_epoch: Optional[int]
    log_every_n_steps: Optional[int]
    accumulate_grad_batches: int
    gradient_clip_algorithm: Optional[str]
    gradient_clip_val: Optional[float]
    early_stopping_patience: Optional[int]
    # decoding
    steps_per_level: List[int]
    maskgit_initial_temp: float


@dataclass
class TransformerModelArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    rope_theta: float


@dataclass
class SoundStormModelArgs(Serializable):
    num_conditioning_categories: int
    num_codec_levels: int
    num_codec_categories_per_level: int
    transformer: TransformerModelArgs
