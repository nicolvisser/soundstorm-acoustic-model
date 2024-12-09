import math
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from soundstorm.config import SoundStormModelArgs
from soundstorm.data import (
    DatasetItem,
    SoundStormTokenizedBatch,
    SoundStormTokenizedItem,
    collate_fn,
)
from soundstorm.model.transformer import TransformerModel
from soundstorm.tokenizer import SoundStormTokenizer


class SoundStormModel(nn.Module):
    def __init__(self, args: SoundStormModelArgs):
        super().__init__()
        self.args = args

        # tokenizer
        self.tokenizer = SoundStormTokenizer(args=args)

        # embeddings
        self.embedding = nn.Embedding(
            num_embeddings=self.tokenizer.global_embedding_size,
            embedding_dim=args.transformer.dim,
            padding_idx=self.tokenizer.MASK_id,
        )

        # transformer
        self.transformer = TransformerModel(args=args.transformer)

        # prediction heads
        self.prediction_heads = torch.nn.ModuleList(
            modules=[
                nn.Linear(args.transformer.dim, args.num_codec_categories_per_level)
                for _ in range(args.num_codec_levels)
            ]
        )

    def _partial_forward(self, batch: SoundStormTokenizedBatch) -> torch.Tensor:
        """
        Sum the embeddings at each time step to get a single embedding per time step
        then pass them through the transformer.
        """
        return self.transformer(
            embeddings=self.embedding(batch.tokens).sum(dim=1),
            q_seqlen=batch.input_lengths,
        )

    def forward(self, batch: SoundStormTokenizedBatch) -> torch.Tensor:
        assert batch.mask_level_idxs is not None
        assert batch.masks is not None
        assert batch.targets is not None

        transformer_output = self._partial_forward(batch)

        # compute the loss over the masked tokens (only)
        loss = 0
        t = 0
        total_masked_tokens = 0
        for input_len, mask_level_idx, mask, target in zip(
            batch.input_lengths, batch.mask_level_idxs, batch.masks, batch.targets
        ):
            # since xformers does not use a batch dimension,
            # we need to slice the transformer output accordingly
            transformer_output_to_use = transformer_output[t : t + input_len]
            # select the prediction head (linear layer) to use
            prediction_head_to_use = self.prediction_heads[mask_level_idx]
            # forward the transformer output through the prediction head
            logits = prediction_head_to_use(transformer_output_to_use)
            # compute the loss over the masked tokens (only)
            masked_logits = logits[mask]
            loss += torch.nn.functional.cross_entropy(
                masked_logits, target, reduction="sum"
            )
            # count the number of masked tokens
            total_masked_tokens += mask.sum().item()
            # update the index
            t += input_len
        # normalize the loss by the number of masked tokens
        loss /= total_masked_tokens

        return loss

    @classmethod
    def from_model_dir(cls, model_dir: str):
        model_dir = Path(model_dir)
        checkpoint_path: Path = model_dir / "best.pth"
        model_args_path: Path = model_dir / "model_args.json"
        assert checkpoint_path.exists(), checkpoint_path
        assert model_args_path.exists(), model_args_path
        model = cls(SoundStormModelArgs.load(model_args_path))
        model.load_state_dict(torch.load(checkpoint_path))
        return model

    @torch.inference_mode()
    def predict(self, item: SoundStormTokenizedItem) -> List[torch.Tensor]:
        """
        Predict logits over the codec categories for each time step at the level
        where the mask was applied.
        """
        assert item.mask_level_idx is not None
        # forward the item through the transformer
        transformer_output = self._partial_forward(collate_fn([item]))
        # select the prediction head (linear layer) to use
        prediction_head_to_use = self.prediction_heads[item.mask_level_idx]
        # forward the transformer output through the prediction head
        # drop the first time step since it corresponds to the SINK token
        logits = prediction_head_to_use(transformer_output)[1:]
        return logits

    @torch.inference_mode()
    def generate(
        self,
        conditioning_tokens: torch.Tensor,
        steps_per_level: List[int],
        maskgit_initial_temp: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate codec tokens for the given conditioning tokens.

        Args:
            conditioning_tokens: (torch.Tensor) The conditioning tokens. Shape: (t,).
            steps_per_level: (List[int]) The number of decoding steps at each RVQ level.
            maskgit_initial_temp: (float) The initial temperature for MaskGIT decoding.
                The temperature is linearly annealed over the decoding steps down to 0.

        Returns:
            (torch.Tensor) The generated codec tokens. Shape: (t, Q).
        """
        assert conditioning_tokens.ndim == 1
        assert isinstance(steps_per_level, list)
        assert isinstance(maskgit_initial_temp, float)
        assert (
            len(steps_per_level) == self.args.num_codec_levels
        ), "The list of steps per level must match the number of RVQ levels."

        t = conditioning_tokens.shape[0]
        Q = self.args.num_codec_levels

        # start with fully masked codec tokens
        # -1 indicates a masked token
        predicted_codec_tokens = torch.full(
            (t, Q), -1, device=conditioning_tokens.device
        )

        # decode one level at a time
        for level_idx in range(Q):

            # repeat for the number of decoding steps at this level
            for step_idx in range(steps_per_level[level_idx]):

                # create a DatasetItem using the conditioning tokens and the predicted codec tokens
                item = DatasetItem(conditioning_tokens, predicted_codec_tokens)
                # get the mask for the current level (i.e, where the predicted codec tokens are still -1)
                masked = predicted_codec_tokens[:, level_idx] == -1  # (T,)
                unmasked = ~masked
                # tokenize the item and specify the mask level index and the mask
                tokenized_item = self.tokenizer.encode(item, level_idx, masked)
                # get the logits over the codec categories for each time step
                logits = self.predict(tokenized_item)  # (T,)

                if step_idx + 1 == steps_per_level[level_idx]:
                    # ============= Greedy decoding at last step of each level =============
                    # simply select the most likely token for each time step
                    predicted_codec_tokens[:, level_idx] = logits.argmax(dim=-1)
                else:  # ======================== MaskGIT decoding =========================
                    # determine the new number of unmasked tokens after decoding
                    u = 0.5 * math.pi * ((step_idx + 1) / steps_per_level[level_idx])
                    p = math.cos(u)
                    new_num_unmasked_tokens = t - int(t * p)

                    # sample a token for each time step
                    # using gumbel sampling, because it's faster than using a softmax
                    temp = maskgit_initial_temp * (
                        step_idx / steps_per_level[level_idx]
                    )
                    sampled_indices = sample_gumbel(logits, temp)

                    # save the indices and values of the unmasked tokens
                    # we will later replace the predicted codec tokens for these time steps
                    # with the previously locked in values
                    # otherwise, decoding will be unstable
                    unmasked_indices = torch.arange(t, device=logits.device)[unmasked]
                    unmasked_values = predicted_codec_tokens[
                        unmasked_indices, level_idx
                    ]

                    # use the sampled probabilities as confidence scores
                    scores = logits[
                        torch.arange(t, device=logits.device), sampled_indices
                    ]  # (T,)
                    # set the confidence of unmasked tokens to 1.0
                    scores.masked_fill_(unmasked, float("inf"))  # (T,)
                    # select the top k confidence scores
                    _, top_indices = torch.topk(scores, new_num_unmasked_tokens)  # (k,)
                    # update the predicted codec tokens and lock them in
                    predicted_codec_tokens[top_indices, level_idx] = sampled_indices[
                        top_indices
                    ]
                    # override with previously locked in values
                    predicted_codec_tokens[unmasked_indices, level_idx] = (
                        unmasked_values
                    )

        return predicted_codec_tokens


def sample_gumbel(logits, temp=1.0):
    """
    Sampling from logits using the Gumbel-max trick.
    """
    noise = torch.zeros_like(logits).uniform_(0, 1).log_().neg_().log_().neg_()
    return ((logits / max(temp, 1e-10)) + noise).argmax(-1)
