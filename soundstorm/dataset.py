from typing import List

import h5py
import torch
from torch.utils.data import Dataset

from soundstorm.config import SoundStormModelArgs
from soundstorm.data import DatasetItem, SoundStormTokenizedItem
from soundstorm.tokenizer import SoundStormTokenizer


class SoundStormTokenizedDataset(Dataset):
    def __init__(
        self,
        args: SoundStormModelArgs,
        conditioning_tokens_dataset_paths: List[str],
        codec_tokens_dataset_paths: List[str],
    ):
        self.args = args

        self.conditioning_tokens_datasets = [
            h5py.File(path, "r") for path in conditioning_tokens_dataset_paths
        ]
        self.codec_tokens_datasets = [
            h5py.File(path, "r") for path in codec_tokens_dataset_paths
        ]

        self.conditioning_tokens_refs = {}
        for conditioning_tokens_dataset in self.conditioning_tokens_datasets:
            for key in conditioning_tokens_dataset.keys():
                self.conditioning_tokens_refs[key] = conditioning_tokens_dataset[key]

        self.codec_tokens_refs = {}
        for codec_tokens_dataset in self.codec_tokens_datasets:
            for key in codec_tokens_dataset.keys():
                self.codec_tokens_refs[key] = codec_tokens_dataset[key]

        self.common_keys = sorted(
            list(
                set(self.conditioning_tokens_refs.keys())
                & set(self.codec_tokens_refs.keys())
            )
        )

        self.tokenizer = SoundStormTokenizer(args)

    def __len__(self):
        return len(self.common_keys)

    def __getitem__(self, index: int) -> SoundStormTokenizedItem:
        # find the key for the item at the given index
        key = self.common_keys[index]
        # fetch the conditioning and codec tokens from the h5 files
        conditioning_tokens = torch.from_numpy(self.conditioning_tokens_refs[key][:])
        codec_tokens = torch.from_numpy(self.codec_tokens_refs[key][:])
        # make sure the conditioning tokens and codec tokens have approximately the same length
        assert abs(conditioning_tokens.shape[0] - codec_tokens.shape[0]) <= 2
        # then truncate the longer one
        minlen = min(conditioning_tokens.shape[0], codec_tokens.shape[0])
        conditioning_tokens = conditioning_tokens[:minlen]
        codec_tokens = codec_tokens[:minlen]
        # now create the item
        item = DatasetItem(
            conditioning_tokens=conditioning_tokens,
            codec_tokens=codec_tokens,
        )
        # then tokenize it while masking randomly
        return self.tokenizer.encode_with_random_mask(item)

    def close(self):
        """Explicitly close all open HDF5 file handles"""
        if hasattr(self, "conditioning_tokens_datasets"):
            for dataset in self.conditioning_tokens_datasets:
                try:
                    dataset.close()
                except Exception:
                    pass

        if hasattr(self, "codec_tokens_datasets"):
            for dataset in self.codec_tokens_datasets:
                try:
                    dataset.close()
                except Exception:
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
