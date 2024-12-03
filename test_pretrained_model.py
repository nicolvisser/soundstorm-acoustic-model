import h5py
import torch
import torchaudio
from speechtokenizer import SpeechTokenizer

from soundstorm.trainer import SoundStormLitModel

from IPython.display import Audio, display


K = 500

path = {
    100: "/mnt/wsl/nvme/code/soundstorm/wandb/run-20241128_141947-gzgrlrun/files",
    500: "/mnt/wsl/nvme/code/soundstorm/wandb/run-20241130_133710-mks9twfx/files",
}

model = SoundStormLitModel.from_model_dir(model_dir=path[K]).cuda()

model.eval()

with h5py.File(
    f"/mnt/wsl/nvme/code/wavlm-quantize-abx/output/layer-11/km-{K}/lmbda-0/units/dev-clean.h5",
    "r",
) as f:
    keys = list(f.keys())
    key = keys[2000]
    conditioning_tokens = f[key][:]

conditioning_tokens = torch.from_numpy(conditioning_tokens).cuda()

codec = model.model.generate(
    conditioning_tokens=conditioning_tokens,
    steps_per_level=[16, 8, 4, 2, 1, 1, 1, 1],
    maskgit_initial_temp=2.0,
)

codec = codec.permute(1, 0).unsqueeze(1)

st = SpeechTokenizer.load_from_checkpoint(
    "checkpoints/speechtokenizer/config.json",
    "checkpoints/speechtokenizer/SpeechTokenizer.pt",
).cuda()

sr = 16000

with torch.no_grad():
    wav = st.decode(codes=codec[:1]).squeeze(0).cpu()
    torchaudio.save("output_first.wav", wav, sr)

with torch.no_grad():
    wav = st.decode(codes=codec[:]).squeeze(0).cpu()
    torchaudio.save("output_all.wav", wav, sr)


display(Audio("output_first.wav"))
display(Audio("output_all.wav"))
