from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from speechtokenizer import SpeechTokenizer
from torch.utils.data import DataLoader

import wandb
from soundstorm.config import SoundStormModelArgs, SoundStormTrainerArgs
from soundstorm.data import SoundStormTokenizedBatch, collate_fn
from soundstorm.dataset import SoundStormTokenizedDataset
from soundstorm.model.soundstorm import SoundStormModel
from soundstorm.scheduler import LinearRampCosineDecayScheduler


class SoundStormLitModel(pl.LightningModule):
    def __init__(
        self, model_args: SoundStormModelArgs, trainer_args: SoundStormTrainerArgs
    ):
        super().__init__()
        self.model_args = model_args
        self.train_args = trainer_args

        self.model = SoundStormModel(model_args)

    def training_step(self, batch: SoundStormTokenizedBatch, batch_idx):
        loss = self.model(batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch: SoundStormTokenizedBatch, batch_idx):
        loss = self.model(batch)
        self.log("val/loss", loss, prog_bar=True, batch_size=batch.batch_size)

        # log audio samples
        MAX_SAMPLES_TO_LOG = 3
        if batch_idx == 0:
            for i, item in zip(range(MAX_SAMPLES_TO_LOG), batch):
                try:
                    conditioning_tokens = self.model.tokenizer.decode(
                        item
                    ).conditioning_tokens.cuda()
                    codes = self.model.generate(
                        conditioning_tokens=conditioning_tokens,
                        steps_per_level=self.train_args.steps_per_level,
                        maskgit_initial_temp=self.train_args.maskgit_initial_temp,
                    )
                    codes = codes.permute(1, 0).unsqueeze(1).cpu()
                    speechtokenizer = self.get_speechtokenizer()
                    wav = speechtokenizer.decode(codes)[0, 0].cpu().numpy()
                    audio = wandb.Audio(wav, sample_rate=16000, caption=f"sample_{i}")
                    self.logger.experiment.log({f"audio_sample_{i}": audio})
                except Exception as e:
                    print(f"Failed to log audio sample {i}: {str(e)}")
                    print(f"Audio array shape: {wav.shape}, dtype: {wav.dtype}")
                    print(f"Audio range: min={wav.min()}, max={wav.max()}")

    def get_speechtokenizer(self):
        if not hasattr(self, "_speechtokenizer"):
            self._speechtokenizer = SpeechTokenizer.load_from_checkpoint(
                "checkpoints/speechtokenizer/config.json",
                "checkpoints/speechtokenizer/SpeechTokenizer.pt",
            ).to("cpu")
        return self._speechtokenizer

    def on_save_checkpoint(self, checkpoint):
        # safely remove speechtokenizer from checkpoint
        if hasattr(self, "_speechtokenizer"):
            if "state_dict" in checkpoint:
                if "_speechtokenizer" in checkpoint["state_dict"]:
                    del checkpoint["state_dict"]["_speechtokenizer"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_args.lr_max,
            betas=self.train_args.betas,
            weight_decay=self.train_args.weight_decay,
        )

        sched_config = {
            "scheduler": LinearRampCosineDecayScheduler(
                optimizer,
                n_linear_steps=self.train_args.warmup_steps,
                n_decay_steps=self.train_args.decay_steps,
                lr_init=0,
                lr_max=self.train_args.lr_max,
                lr_final=self.train_args.lr_final,
            ),
            "frequency": 1,
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": sched_config}

    @classmethod
    def from_model_dir(cls, model_dir: str):
        model_dir = Path(model_dir)
        checkpoint_path: Path = model_dir / "best.ckpt"
        model_args_path: Path = model_dir / "model_args.json"
        train_args_path: Path = model_dir / "trainer_args.json"
        assert checkpoint_path.exists(), checkpoint_path
        assert model_args_path.exists(), model_args_path
        assert train_args_path.exists(), train_args_path
        return cls.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_args=SoundStormModelArgs.load(model_args_path),
            trainer_args=SoundStormTrainerArgs.load(train_args_path),
            strict=False,
        )


def train(model_args: SoundStormModelArgs, trainer_args: SoundStormTrainerArgs):

    torch.set_float32_matmul_precision("medium")  # RTX 3090 speedup

    train_dataset = SoundStormTokenizedDataset(
        args=model_args,
        conditioning_tokens_dataset_paths=trainer_args.conditioning_tokens_train_dataset_paths,
        codec_tokens_dataset_paths=trainer_args.codec_tokens_train_dataset_paths,
    )
    val_dataset = SoundStormTokenizedDataset(
        args=model_args,
        conditioning_tokens_dataset_paths=trainer_args.conditioning_tokens_val_dataset_paths,
        codec_tokens_dataset_paths=trainer_args.codec_tokens_val_dataset_paths,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=trainer_args.batch_size,
        shuffle=True,
        num_workers=trainer_args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=trainer_args.batch_size,
        shuffle=False,
        num_workers=trainer_args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    model = SoundStormLitModel(model_args, trainer_args)

    logger = pl.loggers.WandbLogger(
        log_model=False,
        project="wavlm-discrete-soundstorm-model",
        name=trainer_args.run_name,
    )

    # Get the checkpoint directory path
    checkpoint_dir = Path(logger.experiment.dir)

    # Save the model args and trainer args
    model_args.save(checkpoint_dir / "model_args.json")
    trainer_args.save(checkpoint_dir / "trainer_args.json")

    config = {
        "model_args": model_args.to_dict(),
        "trainer_args": trainer_args.to_dict(),
    }
    logger.log_hyperparams(config)

    print(f"Checkpoint directory: {checkpoint_dir}")

    # Add callbacks for checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        filename="best",
        verbose=True,
    )

    if trainer_args.early_stopping_patience is not None:
        early_stopping_callback = EarlyStopping(
            monitor="val/loss",
            patience=trainer_args.early_stopping_patience,
            mode="min",
            verbose=True,
        )
    else:
        early_stopping_callback = None

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            lr_monitor_callback,
        ],  # Add callbacks here
        accelerator=trainer_args.accelerator,
        fast_dev_run=trainer_args.fast_dev_run,
        max_epochs=trainer_args.max_epochs,
        min_epochs=trainer_args.min_epochs,
        max_steps=trainer_args.max_steps,
        min_steps=trainer_args.min_steps,
        val_check_interval=trainer_args.val_check_interval,
        check_val_every_n_epoch=trainer_args.check_val_every_n_epoch,
        log_every_n_steps=trainer_args.log_every_n_steps,
        accumulate_grad_batches=trainer_args.accumulate_grad_batches,
        gradient_clip_algorithm=trainer_args.gradient_clip_algorithm,
        gradient_clip_val=trainer_args.gradient_clip_val,
    )
    trainer.fit(model, train_loader, val_loader)
