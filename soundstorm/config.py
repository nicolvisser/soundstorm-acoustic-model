from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from simple_parsing import Serializable


@dataclass
class TransformerModelArgs(Serializable):
    """
    Config for the Transformer model.
    """

    dim: int = 1024
    """
    Embedding dimension of the model and the dimension of the transformer layers.
    """
    n_layers: int = 12
    """
    Number of transformer layers.
    """
    hidden_dim: int = 4096
    """
    Hidden dimension of the feed-forward network.
    """
    head_dim: int = 64
    """
    Dimension of each attention head.
    """
    n_heads: int = 16
    """
    Number of attention heads.
    """
    n_kv_heads: int = 16
    """
    Number of key-value heads.
    Can be used to reduce memory usage when n_kv_heads < n_heads.
    """
    norm_eps: float = 1e-6
    """
    Epsilon value for layer normalization.
    """
    rope_theta: float = 1e4
    """
    RoPE theta for relative positional encoding.
    Use larger values for longer sequences.
    """


@dataclass
class SoundStormModelArgs(Serializable):
    """
    Config for the SoundStorm model.
    """

    num_conditioning_categories: int  # required
    """
    Number of conditioning categories
    (i.e., the number of unique units to represent the conditioning signal).
    """
    num_codec_levels: int = 8
    """
    Number of codec levels
    (i.e., the number of RVQ quantizers).
    """
    num_codec_categories_per_level: int = 1024
    """
    Number of codec categories per level
    (i.e., the number of codebook vectors per RVQ quantizer).
    """
    transformer: TransformerModelArgs = field(default_factory=TransformerModelArgs)
    """
    Transformer model config.
    """


@dataclass
class SoundStormTrainerArgs(Serializable):
    """
    Config for the SoundStorm trainer.
    """

    # ------------------------------------ run name ------------------------------------
    run_name: str  # required
    """
    Name to give checkpoint and wandb run.
    """

    # ----------------------------------- dataloader -----------------------------------
    conditioning_tokens_train_dataset_paths: List[str]  # required
    """
    Paths to training conditioning tokens.
    Must be a list of paths to .h5 files.
    Each .h5 file must have a unique key.
    Each key must map to a numpy array of shape (t,), where t is the sequence length.
    Keys must correspond to the keys in the training codec tokens.
    """
    conditioning_tokens_val_dataset_paths: List[str]  # required
    """
    Paths to validation conditioning tokens.
    Must be a list of paths to .h5 files.
    Each .h5 file must have a unique key.
    Each key must map to a numpy array of shape (t,), where t is the sequence length.
    Keys must correspond to the keys in the validation codec tokens.
    """
    codec_tokens_train_dataset_paths: List[str]  # required
    """
    Paths to training codec tokens.
    Must be a list of paths to .h5 files.
    Each .h5 file must have a unique key.
    Each key must map to a numpy array of shape (t,Q), where t is the sequence length.
    Keys must correspond to the keys in the training conditioning tokens.
    """
    codec_tokens_val_dataset_paths: List[str]  # required
    """
    Paths to validation codec tokens.
    Must be a list of paths to .h5 files.
    Each .h5 file must have a unique key.
    Each key must map to a numpy array of shape (t,Q), where t is the sequence length.
    Keys must correspond to the keys in the validation conditioning tokens.
    """
    batch_size: int = 64
    """
    Batch size.
    """
    num_workers: int = 63
    """
    Number of CPU workers for data loading.
    """
    # ------------------------------------ optimizer -----------------------------------
    warmup_steps: int  # required
    """
    Number of steps for learning rate warmup (linear warmup from 0 to lr_max).
    """
    lr_max: float  # required
    """
    Maximum learning rate after warmup.
    """
    decay_steps: int  # required
    """
    Number of steps for learning rate decay (cosine decay from lr_max to lr_final).
    """
    lr_final: float  # required
    """
    Final learning rate after decay (will stay constant after warmup_steps + decay_steps).
    """
    betas: Tuple[float, float] = (0.9, 0.999)
    """
    Beta values for AdamW optimizer.
    """
    weight_decay: float = 0.01
    """
    Weight decay for AdamW optimizer.
    """
    # ----------------------------------- pl.trainer -----------------------------------
    accelerator: str = "gpu"
    """
    See Lightning Trainer.
    """
    strategy: str = "ddp"
    """
    See Lightning Trainer.
    """
    devices: int = 2
    """
    See Lightning Trainer.
    """
    precision: str = "bf16-mixed"
    """
    See Lightning Trainer.
    """
    fast_dev_run: bool = False
    """
    See Lightning Trainer.
    """
    max_steps: int = -1
    """
    See Lightning Trainer.
    """
    val_check_interval: float = 1.0
    """
    See Lightning Trainer.
    """
    check_val_every_n_epoch: int = 1
    """
    See Lightning Trainer.
    """
    log_every_n_steps: int = 1
    """
    See Lightning Trainer.
    """
    accumulate_grad_batches: int = 1
    """
    See Lightning Trainer.
    """
    gradient_clip_algorithm: Optional[str] = "value"
    """
    See Lightning Trainer.
    """
    gradient_clip_val: Optional[float] = 1.0
    """
    See Lightning Trainer.
    """
    early_stopping_patience: Optional[int] = 4
    """
    See Lightning Trainer.
    """
    # -------------------------------- logging samples ---------------------------------
    num_samples_to_log: int = 8
    """
    Number of audio samples to log from the first batch of the validation set.
    Set to 0 if you don't want to log any samples.
    Then you don't need to specify the other logging parameters.
    """
    steps_per_level: Optional[List[int]] = [16, 8, 4, 2, 1, 1, 1, 1]
    """
    Number of MaskGIT decoding steps per RVQ level when logging samples.
    See SoundStorm paper for details.
    If 1, then greedy decoding is used.
    Greedy decoding is always used for the last level.
    """
    maskgit_initial_temp: Optional[float] = 1.0
    """
    Initial temperature for MaskGIT decoding when logging samples.
    If the number of steps per level is greater than 1,
    then we start sampling (gumbel) from this temperature;
    then, we anneal from this temperature to 0 at the last step in the level.
    """
    speechtokenizer_config_path: Optional[str] = None
    """
    Path to SpeechTokenizer config for logging samples.
    This model is used to convert output of the model to a waveform.
    """
    speechtokenizer_checkpoint_path: Optional[str] = None
    """
    Path to SpeechTokenizer checkpoint for logging samples.
    This model is used to convert output of the model to a waveform.
    """

    def __post_init__(self):
        if self.num_samples_to_log > 0 and (
            self.steps_per_level is None
            or self.maskgit_initial_temp is None
            or self.speechtokenizer_config_path is None
            or self.speechtokenizer_checkpoint_path is None
        ):
            raise ValueError(
                "steps_per_level, maskgit_initial_temp, speechtokenizer_config_path, and speechtokenizer_checkpoint_path must be specified if num_samples_to_log > 0"
            )
