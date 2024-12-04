from soundstorm.config import (
    SoundStormTrainerArgs,
    SoundStormModelArgs,
    TransformerModelArgs,
)
from soundstorm.trainer import train

model_args = SoundStormModelArgs(
    num_conditioning_categories=100,
    num_codec_levels=8,
    num_codec_categories_per_level=1024,
    transformer=TransformerModelArgs(
        dim=1024,
        n_layers=12,
        head_dim=64,
        hidden_dim=4096,
        n_heads=16,
        n_kv_heads=16,
        norm_eps=1e-6,
        rope_theta=1e4,
    ),
)

trainer_args = SoundStormTrainerArgs(
    # run name
    run_name=f"layer-11-km100-lmbda-0-ddp",
    # dataloader
    conditioning_tokens_train_dataset_paths=[
        f"/workspace/data/units/train-clean-100.h5",
    ],
    conditioning_tokens_val_dataset_paths=[
        f"/workspace/data/units/dev-clean.h5",
    ],
    codec_tokens_train_dataset_paths=[
        "/workspace/data/speechtokenizer/train-clean-100.h5",
    ],
    codec_tokens_val_dataset_paths=[
        "/workspace/data/speechtokenizer/dev-clean.h5",
    ],
    batch_size=6,
    num_workers=63,
    # optimizer
    lr_max=1e-4,
    warmup_steps=4500,
    decay_steps=4500 * 9,
    lr_final=5e-6,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    # pl.trainer
    accelerator="gpu",
    precision="bf16-mixed",
    strategy="ddp",
    devices=2,
    fast_dev_run=False,
    max_epochs=-1,
    min_epochs=None,
    max_steps=-1,
    min_steps=None,
    val_check_interval=0.25,
    check_val_every_n_epoch=1,
    log_every_n_steps=1,
    accumulate_grad_batches=1,
    gradient_clip_algorithm="value",
    gradient_clip_val=1.0,
    early_stopping_patience=4,
    # decoding
    steps_per_level=[16, 8, 4, 2, 1, 1, 1, 1],
    maskgit_initial_temp=3.0,
    # log samples
    num_samples_to_log=3,
    speechtokenizer_config_path="/workspace/data/speechtokenizer/speechtokenizer_hubert_avg_config.json",
    speechtokenizer_checkpoint_path="/workspace/data/speechtokenizer/SpeechTokenizer.pt",
)


train(model_args, trainer_args)
