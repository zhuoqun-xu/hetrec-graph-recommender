"""Movie-side meta-path GraphSAGE-style encoder with a BPR recommendation head.

This is a transductive warm-item adaptation of the thesis idea: movie ID
embeddings are learned, users have learned ID embeddings, and semantic attention
is per movie rather than the thesis's graph-level path weighting.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from graph_data import SampledPath


class MetaPathRecommender(nn.Module):
    def __init__(
        self,
        user_count: int,
        item_count: int,
        paths: dict[str, SampledPath],
        dimensions: int = 32,
    ) -> None:
        super().__init__()
        if not paths:
            raise ValueError("At least one meta-path is required")
        self.path_names = tuple(paths)
        self.user_embedding = nn.Embedding(user_count, dimensions)
        self.movie_embedding = nn.Embedding(item_count, dimensions)
        self.movie_bias = nn.Embedding(item_count, 1)
        self.self_projection = nn.Linear(dimensions, dimensions)
        self.path_projection = nn.ModuleDict(
            {name: nn.Linear(2 * dimensions, dimensions) for name in self.path_names}
        )
        self.attention_hidden = nn.Linear(dimensions, max(8, dimensions // 2))
        self.attention_output = nn.Linear(max(8, dimensions // 2), 1)

        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.movie_embedding.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.movie_bias.weight)
        for name, sampled in paths.items():
            if sampled.indices.shape[0] != item_count:
                raise ValueError(f"Path {name} has wrong movie count")
            self.register_buffer(
                f"{name}_indices",
                torch.from_numpy(sampled.indices.astype(np.int64)),
            )
            self.register_buffer(
                f"{name}_mask",
                torch.from_numpy(sampled.mask.copy()),
            )

    def encode_movies(
        self,
        enabled_paths: set[str] | None = None,
        equal_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if enabled_paths is None:
            enabled_paths = set(self.path_names)
        unknown = enabled_paths - set(self.path_names)
        if unknown:
            raise ValueError(f"Unknown enabled paths: {sorted(unknown)}")

        base = self.movie_embedding.weight
        representations = [
            F.normalize(F.relu(self.self_projection(base)), p=2, dim=1)
        ]
        available = [torch.ones(base.shape[0], dtype=torch.bool, device=base.device)]
        for name in self.path_names:
            indices = getattr(self, f"{name}_indices")
            mask = getattr(self, f"{name}_mask")
            gathered = base[indices]
            neighbor_sum = (gathered * mask.unsqueeze(-1)).sum(dim=1)
            neighbor_count = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            neighbor_mean = neighbor_sum / neighbor_count
            path_representation = F.normalize(
                F.relu(
                    self.path_projection[name](torch.cat([base, neighbor_mean], dim=1))
                ),
                p=2,
                dim=1,
            )
            representations.append(path_representation)
            available.append(mask.any(dim=1) & (name in enabled_paths))

        stack = torch.stack(representations, dim=1)  # movies x branches x dim
        availability = torch.stack(available, dim=1)
        if equal_attention:
            weights = availability.float()
            weights = weights / weights.sum(dim=1, keepdim=True)
        else:
            logits = self.attention_output(
                torch.tanh(self.attention_hidden(stack))
            ).squeeze(-1)
            logits = logits.masked_fill(~availability, -1e9)
            weights = torch.softmax(logits, dim=1)
        movie_vectors = (weights.unsqueeze(-1) * stack).sum(dim=1)
        return movie_vectors, weights

    def score_pairs(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        movie_vectors: torch.Tensor,
    ) -> torch.Tensor:
        users = self.user_embedding(user_indices)
        movies = movie_vectors[item_indices]
        return (users * movies).sum(dim=1) + self.movie_bias(item_indices).squeeze(-1)

    @torch.no_grad()
    def full_score_matrix(
        self,
        enabled_paths: set[str] | None = None,
        equal_attention: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        movies, weights = self.encode_movies(enabled_paths, equal_attention)
        scores = (
            self.user_embedding.weight @ movies.T
            + self.movie_bias.weight.squeeze(-1)[None, :]
        )
        return scores.detach().cpu().numpy(), weights.detach().cpu().numpy()
