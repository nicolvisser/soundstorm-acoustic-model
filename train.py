import click

from soundstorm.config import (
    SoundStormModelArgs,
    SoundStormTrainerArgs,
    TransformerModelArgs,
)
from soundstorm.trainer import train


@click.command()
@click.option("--k", required=True, type=int, help="Number of clusters (K)")
@click.option("--lmbda", required=True, type=int, help="Lambda parameter")
def main(k: int, lmbda: float):
    model_args = SoundStormModelArgs(
        num_conditioning_categories=k,
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
        run_name=f"km-{k}-lmbda-{lmbda}-ddp",
        conditioning_tokens_train_dataset_paths=[
            f"/workspace/data/units/wavlm/layer-11/km-{k}/lmbda-{lmbda}/train-clean-100.h5",
            f"/workspace/data/units/wavlm/layer-11/km-{k}/lmbda-{lmbda}/train-clean-360.h5",
        ],
        conditioning_tokens_val_dataset_paths=[
            f"/workspace/data/units/wavlm/layer-11/km-{k}/lmbda-{lmbda}/dev-clean.h5",
        ],
        codec_tokens_train_dataset_paths=[
            f"/workspace/data/speechtokenizer/train-clean-100.h5",
            f"/workspace/data/speechtokenizer/train-clean-360.h5",
        ],
        codec_tokens_val_dataset_paths=[
            f"/workspace/data/speechtokenizer/dev-clean.h5",
        ],
        batch_size=64,
        num_workers=63,
        lr_max=1e-4,
        warmup_steps=1000,
        decay_steps=19000,
        lr_final=1e-6,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        accelerator="gpu",
        precision="bf16-mixed",
        strategy="ddp",
        devices=2,
        fast_dev_run=False,
        max_epochs=-1,
        min_epochs=None,
        max_steps=-1,
        min_steps=None,
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        accumulate_grad_batches=1,
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
        early_stopping_patience=4,
        steps_per_level=[16, 8, 4, 2, 1, 1, 1, 1],
        maskgit_initial_temp=3.0,
        num_samples_to_log=3,
        speechtokenizer_config_path="/workspace/data/speechtokenizer/speechtokenizer_hubert_avg_config.json",
        speechtokenizer_checkpoint_path="/workspace/data/speechtokenizer/SpeechTokenizer.pt",
    )

    train(model_args, trainer_args)


if __name__ == "__main__":
    main()
