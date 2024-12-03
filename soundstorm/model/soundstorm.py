import math
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

    def partial_forward(self, batch: SoundStormTokenizedBatch) -> torch.Tensor:
        return self.transformer(
            embeddings=self.embedding(batch.tokens).sum(dim=1),
            q_seqlen=batch.input_lengths,
        )

    def forward(self, batch: SoundStormTokenizedBatch) -> torch.Tensor:
        assert batch.mask_level_idxs is not None
        assert batch.masks is not None
        assert batch.targets is not None

        transformer_output = self.partial_forward(batch)

        loss = 0
        t = 0
        total_masked_tokens = 0
        for input_len, mask_level_idx, mask, target in zip(
            batch.input_lengths, batch.mask_level_idxs, batch.masks, batch.targets
        ):
            transformer_output_to_use = transformer_output[t : t + input_len]
            prediction_head_to_use = self.prediction_heads[mask_level_idx]
            logits = prediction_head_to_use(transformer_output_to_use)
            masked_logits = logits[mask]
            loss += torch.nn.functional.cross_entropy(
                masked_logits, target, reduction="sum"
            )
            total_masked_tokens += mask.sum().item()
            t += input_len
        loss /= total_masked_tokens

        return loss

    @torch.inference_mode()
    def predict(self, item: SoundStormTokenizedItem) -> List[torch.Tensor]:
        assert item.mask_level_idx is not None
        transformer_output = self.partial_forward(collate_fn([item]))
        prediction_head_to_use = self.prediction_heads[item.mask_level_idx]
        logits = prediction_head_to_use(transformer_output)[1:]
        return logits

    @torch.inference_mode()
    def generate(
        self,
        conditioning_tokens: torch.Tensor,
        steps_per_level: List[int],
        maskgit_initial_temp: float = 1.0,
    ) -> torch.Tensor:
        assert conditioning_tokens.ndim == 1
        assert len(steps_per_level) == self.args.num_codec_levels

        T = conditioning_tokens.shape[0]
        Q = self.args.num_codec_levels

        predicted_codec_tokens = torch.full(
            (T, Q), -1, device=conditioning_tokens.device
        )

        for level_idx in range(Q):

            for step_idx in range(steps_per_level[level_idx]):

                item = DatasetItem(conditioning_tokens, predicted_codec_tokens)
                masked = predicted_codec_tokens[:, level_idx] == -1  # (T,)
                unmasked = ~masked
                tokenized_item = self.tokenizer.encode(item, level_idx, masked)
                logits = self.predict(tokenized_item)  # (T,)

                if step_idx + 1 == steps_per_level[level_idx]:
                    # ======================== greedy decoding ========================
                    predicted_codec_tokens[:, level_idx] = logits.argmax(dim=-1)
                else:
                    # ======================== MaskGIT decoding =======================
                    # determine the new number of unmasked tokens after decoding
                    u = 0.5 * math.pi * ((step_idx + 1) / steps_per_level[level_idx])
                    p = math.cos(u)
                    new_num_unmasked_tokens = T - int(T * p)

                    # sample a token for each time step
                    temp = maskgit_initial_temp * (
                        step_idx / steps_per_level[level_idx]
                    )
                    sampled_indices = sample_gumbel(logits, temp)

                    unmasked_indices = torch.arange(T, device=logits.device)[unmasked]
                    unmasked_values = predicted_codec_tokens[
                        unmasked_indices, level_idx
                    ]

                    # use the sampled probabilities as confidence scores
                    scores = logits[
                        torch.arange(T, device=logits.device), sampled_indices
                    ]  # (T,)
                    # set the confidence of unmasked tokens to 1.0
                    scores.masked_fill_(unmasked, float("inf"))  # (T,)
                    # select the top k confidence scores
                    _, top_indices = torch.topk(scores, new_num_unmasked_tokens)  # (k,)
                    # update the predicted codec tokens and lock them in
                    predicted_codec_tokens[top_indices, level_idx] = sampled_indices[
                        top_indices
                    ]
                    predicted_codec_tokens[unmasked_indices, level_idx] = (
                        unmasked_values
                    )

        return predicted_codec_tokens


def sample_gumbel(logits, temp=1.0):
    noise = torch.zeros_like(logits).uniform_(0, 1).log_().neg_().log_().neg_()
    return ((logits / max(temp, 1e-10)) + noise).argmax(-1)
