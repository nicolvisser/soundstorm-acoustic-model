from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch


@dataclass
class DatasetItem:
    conditioning_tokens: torch.Tensor  # (T,)
    codec_tokens: torch.Tensor  # (T,Q)

    def __post_init__(self):
        assert self.conditioning_tokens.ndim == 1
        assert self.codec_tokens.ndim == 2
        assert self.conditioning_tokens.shape[0] == self.codec_tokens.shape[0]

    @property
    def T(self):
        return self.conditioning_tokens.shape[0]

    @property
    def Q(self):
        return self.codec_tokens.shape[1]


@dataclass
class SoundStormTokenizedItem:
    tokens: torch.Tensor  # (T+1, Q+1)
    mask_level_idx: int
    mask: torch.Tensor
    target: Optional[torch.Tensor]

    def __post_init__(self):
        assert self.tokens.ndim == 2

        if self.mask is not None and self.target is not None:
            assert self.mask.sum() == len(
                self.target
            ), "There should be exactly one target for every true item in the mask"


@dataclass
class SoundStormTokenizedBatch:
    tokens: torch.Tensor  # (N,Q+1)
    input_lengths: List[int]  # len B
    mask_level_idxs: List[int]  # len B
    masks: List[torch.Tensor]  # len B
    targets: Optional[List[torch.Tensor]]  # len B

    def __post_init__(self):
        assert self.tokens.ndim == 2
        assert isinstance(self.input_lengths, list)
        assert self.tokens.shape[0] == sum(self.input_lengths)

    @property
    def batch_size(self):
        return len(self.input_lengths)

    @property
    def total_sequence_length(self):
        return self.tokens.shape[0]

    def __iter__(self) -> Iterator[SoundStormTokenizedItem]:
        start_idx = 0
        # Create empty iterables for None values
        targets = self.targets or [None] * self.batch_size

        for input_length, mask_level_idx, mask, target in zip(
            self.input_lengths, self.mask_level_idxs, self.masks, targets
        ):
            yield SoundStormTokenizedItem(
                tokens=self.tokens[start_idx : start_idx + input_length],
                mask_level_idx=mask_level_idx,
                mask=mask,
                target=target,
            )
            start_idx += input_length

    def to(self, device: torch.device):
        return SoundStormTokenizedBatch(
            tokens=self.tokens.to(device),
            input_lengths=self.input_lengths,
            mask_level_idxs=self.mask_level_idxs,
            masks=[mask.to(device) for mask in self.masks],
            targets=(
                [target.to(device) for target in self.targets]
                if self.targets is not None
                else None
            ),
        )


def collate_fn(items: List[SoundStormTokenizedItem]) -> SoundStormTokenizedBatch:
    return SoundStormTokenizedBatch(
        # concatenate the tokens along the time dimension
        tokens=torch.cat([item.tokens for item in items]),
        # specify the length of each item in the batch
        input_lengths=[item.tokens.shape[0] for item in items],
        # return the rest as lists
        mask_level_idxs=[item.mask_level_idx for item in items],
        masks=[item.mask for item in items],
        targets=[item.target for item in items],
    )
