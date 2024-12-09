from soundstorm.trainer import SoundStormLitModel
from soundstorm.config import SoundStormModelArgs, SoundStormTrainerArgs
from pathlib import Path
import click
import torch

@click.command()
@click.argument("ckpt_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("pth_dir", type=click.Path(exists=False, file_okay=False, path_type=Path))
def ckpt2pth(ckpt_dir: Path, pth_dir: Path):
    lit_model: SoundStormLitModel = SoundStormLitModel.from_model_dir(ckpt_dir)
    model = lit_model.model
    # save model state dict
    pth_path = pth_dir / "best.pth"
    pth_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), pth_path)
    # save model config
    config_path = pth_dir / "model_args.json"
    args: SoundStormModelArgs = lit_model.model_args
    args.save(config_path)
    # save trainer config
    trainer_path = pth_dir / "trainer_args.json"
    trainer_args: SoundStormTrainerArgs = lit_model.train_args
    trainer_args.save(trainer_path)


if __name__ == "__main__":
    ckpt2pth()
