"""
This module contains a tokenizer that should be applied to the data before training.
The tokenizer applies the SoundStorm masking strategy and prepends a SINK token.
We do this here to move some computation to the CPU during data loading, instead of the GPU.
"""

import math
import random

import numpy as np
import torch

from soundstorm.config import SoundStormModelArgs
from soundstorm.data import DatasetItem, SoundStormTokenizedItem


class SoundStormTokenizer:
    """
    Tokenizer that applies the SoundStorm masking strategy and prepends a SINK token.

    See SoundStorm paper for details about the masking strategy.

    Args:
        - args: SoundStormModelArgs.

    Methods:
        - encode: Encode a DatasetItem into a SoundStormTokenizedItem. Manually specify the mask. Used for inference.
        - encode_with_random_mask: Encode a DatasetItem into a SoundStormTokenizedItem with a random mask. Used for training.
        - decode: Decode a SoundStormTokenizedItem into a DatasetItem.
    """

    def __init__(self, args: SoundStormModelArgs):
        self.args = args

        self.MASK_id = 0
        """
        ID of the MASK token. Used by the model to zero out the embedding for these tokens before summing across RVQ levels.
        """
        self.SINK_id = 1
        """
        ID of the SINK (start) token. Prepended to the tokens before passing them to the model.
        """
        self.SINK_column = torch.full(
            (1, args.num_codec_levels + 1), self.SINK_id, dtype=torch.long
        )

        self.conditioning_offset = 2
        """
        Offset added to the conditioning tokens.
        We use a global embedding for all tokens (special, conditioning, codec layer 1, ... ).
        Therefore, we add offsets to the IDs to avoid collisions.
        """

        self.codec_offsets = np.cumsum(
            [2 + args.num_conditioning_categories]
            + [args.num_codec_categories_per_level] * args.num_codec_levels
        )
        """
        Offsets for the codec tokens.
        We use a global embedding for all tokens (special, conditioning, codec layer 1, ... ).
        Therefore, we add offsets to the IDs to avoid collisions.
        """
        self.codec_offsets = self.codec_offsets[None, :-1]  # (1, Q)
        self.codec_offsets = torch.tensor(self.codec_offsets, dtype=torch.long)

    def encode_with_random_mask(self, item: DatasetItem) -> SoundStormTokenizedItem:
        """
        Encode a DatasetItem into a SoundStormTokenizedItem with a random mask.
        See SoundStorm paper for details about the masking strategy.
        """

        # random mask level
        mask_level_idx = random.randint(0, self.args.num_codec_levels - 1)
        # random mask
        u = torch.rand(1) * math.pi / 2
        p = torch.cos(u)
        mask = torch.rand(item.t) < p
        return self.encode(
            item, mask_level_idx=mask_level_idx, mask=mask, create_target=True
        )

    def encode(
        self,
        item: DatasetItem,
        mask_level_idx: int,
        mask: torch.Tensor,
        create_target: bool = False,
    ) -> SoundStormTokenizedItem:
        """
        Encode a DatasetItem into a SoundStormTokenizedItem where you can manually specify the mask.
        Used for inference. See SoundStorm paper for details about the masking strategy.

        Args:
            - item: DatasetItem.
            - mask_level_idx: Level at which to apply the mask.
            - mask: Mask to apply to the tokens at the specified level.
            - create_target: Whether to create a target for the loss.
                The target is a tensor with the elements that were masked at the specified level.
        """

        assert (
            mask.ndim == 1
        ), "Mask must be 1D. You should only provide a mask for 'mask_layer_idx'. All higher levels are masked automatically and lower levels are unmasked."
        assert (
            mask.shape[0] == item.t
        ), f"Mask must have the same length as the input. Input length: {item.t}. Mask length: {mask.shape[0]}."

        # ----- add offsets for global nn.Embedding -----

        conditioning_tokens = item.conditioning_tokens + self.conditioning_offset
        self.codec_offsets = self.codec_offsets.to(conditioning_tokens.device)
        codec_tokens = item.codec_tokens + self.codec_offsets

        # ----- apply mask -----

        # mask at the specified level
        codec_tokens[mask, mask_level_idx] = self.MASK_id
        # mask all higher levels
        codec_tokens[:, mask_level_idx + 1 :] = self.MASK_id

        # get target tokens
        target = item.codec_tokens[mask, mask_level_idx] if create_target else None

        # ----- construct tokens -----

        # combine conditioning and codec tokens
        tokens = torch.cat([conditioning_tokens.unsqueeze(1), codec_tokens], dim=1)
        # prepend SINK tokens
        self.SINK_column = self.SINK_column.to(tokens.device)
        tokens = torch.cat([self.SINK_column, tokens], dim=0)
        # prepend False to mask
        mask = torch.cat([torch.tensor([False], device=mask.device), mask], dim=0)

        return SoundStormTokenizedItem(
            tokens=tokens,
            mask_level_idx=mask_level_idx,
            mask=mask,
            target=target,
        )

    def decode(self, item: SoundStormTokenizedItem) -> DatasetItem:
        """
        Decode a SoundStormTokenizedItem into a DatasetItem.
        """
        # remove first timestep with SINK tokens:
        tokens = item.tokens[1:]
        # split into conditioning and codec tokens:
        conditioning_tokens = tokens[:, :1]
        conditioning_tokens = conditioning_tokens - self.conditioning_offset
        codec_tokens = tokens[:, 1:]
        # decode conditioning tokens:
        masks = codec_tokens == self.MASK_id
        self.codec_offsets = self.codec_offsets.to(codec_tokens.device)
        codec_tokens = codec_tokens - self.codec_offsets
        codec_tokens.masked_fill_(masks, -1)
        return DatasetItem(
            conditioning_tokens=conditioning_tokens.squeeze(1),
            codec_tokens=codec_tokens,
        )

    @property
    def global_embedding_size(self) -> int:
        """
        Size of the global embedding (num_special_tokens + num_conditioning_categories + num_codec_categories).
        """
        return (
            2
            + self.args.num_conditioning_categories
            + self.args.num_codec_categories_per_level * self.args.num_codec_levels
        )
