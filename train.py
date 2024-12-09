import click

from soundstorm.config import SoundStormModelArgs, SoundStormTrainerArgs
from soundstorm.trainer import train


@click.command()
@click.option("--k", required=True, type=int, help="Number of clusters (K)")
@click.option("--lmbda", required=True, type=int, help="Lambda parameter")
def main(k: int, lmbda: float):
    model_args = SoundStormModelArgs(
        num_conditioning_categories=k,
        # use default values for the rest, see config.py
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
        batch_size=64,  # no benefit when exactly 2**n when using xformers
        num_workers=0,
        warmup_steps=1000,
        lr_max=1e-4,
        decay_steps=9000,
        lr_final=1e-6,
        num_samples_to_log=8,
        steps_per_level=[16, 8, 4, 2, 1, 1, 1, 1],
        maskgit_initial_temp=1.0,
        speechtokenizer_config_path="/workspace/data/speechtokenizer/speechtokenizer_hubert_avg_config.json",
        speechtokenizer_checkpoint_path="/workspace/data/speechtokenizer/SpeechTokenizer.pt",
        val_check_interval=0.5,
        # use default values for the rest, see config.py
    )

    train(model_args, trainer_args)


if __name__ == "__main__":
    main()
