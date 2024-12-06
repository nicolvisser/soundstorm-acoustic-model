"""
This module defines the core data structures with checks to ensure correct format:

1. DatasetItem:
   - Output of the torch.utils.data.Dataset.

2. SoundStormTokenizedItem:
   - Output of the tokenizer which applies the SoundStorm masking strategy and adds a SINK token.

3. SoundStormTokenizedBatch:
   - Output of the collate_fn (i.e output of the torch.utils.data.DataLoader).

4. collate_fn:
   - Function that takes a list of SoundStormTokenizedItem and returns a SoundStormTokenizedBatch.
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch


@dataclass
class DatasetItem:
    """
    A dataclass representing a dataset item, which is a pair of conditioning and codec tokens.

    Attributes:
        conditioning_tokens (torch.Tensor): A tensor of shape (t,) representing the conditioning tokens.
        codec_tokens (torch.Tensor): A tensor of shape (t,Q) representing the codec tokens.

    The conditioning tokens are the input tokens to the SoundStorm acoustic model.
    These can be the cluster IDs from an SSL model, e.g. WavLM.
    The codec tokens are the output tokens of the SoundStorm acoustic model.
    These can be the discrete codes from any RVQ-based audio codec, e.g. SpeechTokenizer.
    The time dimension t must be the same for both conditioning and codec tokens.
    Both WavLM and SpeechTokenizer uses codes at 50 Hz.
    Q is the number of RVQ levels used by the codec.
    For SpeechTokenizer, Q = 8.
    """

    conditioning_tokens: torch.Tensor  # (t,)
    codec_tokens: torch.Tensor  # (t,Q)

    def __post_init__(self):
        assert self.conditioning_tokens.ndim == 1  # 1D
        assert self.codec_tokens.ndim == 2  # 2D
        assert self.conditioning_tokens.shape[0] == self.codec_tokens.shape[0]  # t = t

    @property
    def t(self):
        """
        The sequence length of the conditioning and codec tokens.
        """
        return self.conditioning_tokens.shape[0]

    @property
    def Q(self):
        """
        The number of RVQ levels used by the codec.
        """
        return self.codec_tokens.shape[1]


@dataclass
class SoundStormTokenizedItem:
    """
    A dataclass representing a tokenized item for SoundStorm.
    While the DatasetItem is unmasked, the SoundStormTokenizedItem includes a mask.

    Attributes:
        tokens (torch.Tensor): A tensor of shape (t+1, Q+1) representing the conditioning and codec tokens.
        mask_level_idx (int): An index indicating the level of masking.
        mask (torch.Tensor): A tensor representing the mask.
        target (Optional[torch.Tensor]): An optional tensor representing the target.

    The tokens are the concatenation of the conditioning and codec tokens.
    In the time dimension, a SINK token is prepended to the conditioning tokens.
    In the last dimension, the first index corresponds to the conditioning tokens and the remaining indices correspond to the codec tokens.
    The codec tokens will be masked with the SoundStorm masking strategy.
    The SoundStorm masking strategy is defined by an index of the level to mask and a mask for that specific level.
    Optionally, a target can be provided, these are the correct codec tokens for the masked items at the given level.
    The target is only needed for training and not during inference.
    """

    tokens: torch.Tensor  # (1+t, 1+Q)
    mask_level_idx: int  # between 1 and Q-1
    mask: torch.Tensor  # (1+t,)
    target: Optional[torch.Tensor]  # (m,)

    def __post_init__(self):
        assert self.tokens.ndim == 2  # 2D
        assert isinstance(self.mask_level_idx, int)
        assert 1 <= self.mask_level_idx <= self.Q - 1
        assert self.mask.ndim == 1  # 1D
        assert self.target is None or self.target.ndim == 1  # 1D
        assert self.target is None or self.mask.sum() == len(
            self.target
        ), "There should be exactly one target for every true item in the mask"


@dataclass
class SoundStormTokenizedBatch:
    """
    A dataclass representing a batch of tokenized items for SoundStorm.

    Attributes:
        tokens (torch.Tensor): A tensor of shape (N,Q+1) representing the individual tokens concatenated along the time dimension.
        input_lengths (List[int]): A list of lengths of the individual input sequences.
        mask_level_idxs (List[int]): A list of indices indicating the level of masking for each item.
        masks (List[torch.Tensor]): A list of masks for each item.
        targets (Optional[List[torch.Tensor]]): A list of targets for each item.

        Our SoundStorm implementation uses xformers to improve training efficiency.
        The xformers memory_efficient_attention routine requires that the input items be concatenated along the time dimension.
        Then a BlockDiagonalMask is applied as the attention bias based on the input lengths.
        This avoids the need to pad the input to the maximum sequence length when using a separate batch dimension.
        The SoundStormTokenizedBatch class includes the masking and targets for each item in the batch.
    """

    tokens: torch.Tensor  # (N,Q+1)
    input_lengths: List[int]  # len B
    mask_level_idxs: List[int]  # len B
    masks: List[torch.Tensor]  # len B
    targets: Optional[List[torch.Tensor]]  # len B

    def __post_init__(self):
        assert self.tokens.ndim == 2  # 2D
        assert isinstance(self.input_lengths, list)
        assert isinstance(self.mask_level_idxs, list)
        assert isinstance(self.masks, list)
        assert self.targets is None or isinstance(self.targets, list)
        assert self.tokens.shape[0] == sum(self.input_lengths)  # N=\sum_{i=1}^{B}{t_i}

    @property
    def batch_size(self):
        """
        The number of items in the batch.
        """
        return len(self.input_lengths)

    @property
    def total_sequence_length(self):
        """
        The total sequence length of the concatenated items in the batch.
        """
        return self.tokens.shape[0]

    def __iter__(self) -> Iterator[SoundStormTokenizedItem]:
        """
        Iterate over the items in the batch and yield SoundStormTokenizedItem.
        """
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
        """
        Move the tensors in the batch to a specific device.
        """
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
    """
    Collate a list of SoundStormTokenizedItem into a SoundStormTokenizedBatch.

    The xformers memory_efficient_attention routine requires that the input items be concatenated along the time dimension.
    Then a BlockDiagonalMask is applied as the attention bias based on the input lengths.
    This avoids the need to pad the input to the maximum sequence length when using a separate batch dimension.
    """
    return SoundStormTokenizedBatch(
        # concatenate the tokens along the time dimension
        tokens=torch.cat([item.tokens for item in items], dim=0),
        # specify the length of each item in the batch
        input_lengths=[item.tokens.shape[0] for item in items],
        # return the rest as lists
        mask_level_idxs=[item.mask_level_idx for item in items],
        masks=[item.mask for item in items],
        targets=[item.target for item in items],
    )
