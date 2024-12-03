import math
import random
from typing import List

import numpy as np
import torch

from soundstorm.config import SoundStormModelArgs
from soundstorm.data import DatasetItem, SoundStormTokenizedItem


class SoundStormTokenizer:
    def __init__(self, args: SoundStormModelArgs):
        self.args = args

        self.MASK_id = 0
        self.SINK_id = 1

        self.SINK_column = torch.full(
            (1, args.num_codec_levels + 1), self.SINK_id, dtype=torch.long
        )

        self.conditioning_offset = 2

        self.codec_offsets = np.cumsum(
            [2 + args.num_conditioning_categories]
            + [args.num_codec_categories_per_level] * args.num_codec_levels
        )
        self.codec_offsets = self.codec_offsets[None, :-1]  # (1, Q)
        self.codec_offsets = torch.tensor(self.codec_offsets, dtype=torch.long)

    def encode_with_random_mask(self, item: DatasetItem) -> SoundStormTokenizedItem:
        # random mask level
        mask_level_idx = random.randint(0, self.args.num_codec_levels - 1)
        # random mask
        u = torch.rand(1) * math.pi / 2
        p = torch.cos(u)
        mask = torch.rand(item.T) < p
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

        assert (
            mask.ndim == 1
        ), "Mask must be 1D. You should only provide a mask for 'mask_layer_idx'. All higher levels are masked automatically."
        assert (
            mask.shape[0] == item.T
        ), f"Mask must have the same length as the input. Input length: {item.T}. Mask length: {mask.shape[0]}."

        # ----- add offsets for global nn.embedding -----

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
        return (
            2
            + self.args.num_conditioning_categories
            + self.args.num_codec_categories_per_level * self.args.num_codec_levels
        )
