"""Small invariant checks; run with python -B -m unittest."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.nn import functional as F

from graph_data import SampledPath, shuffle_sampled_endpoints
from meta_path_model import MetaPathRecommender


class MetaPathTests(unittest.TestCase):
    def make_path(self) -> SampledPath:
        indices = np.asarray(
            [[1, 2], [0, 2], [0, 1], [0, 1], [0, 1]], dtype=np.int32
        )
        mask = np.asarray(
            [[1, 1], [1, 1], [1, 1], [1, 1], [0, 0]], dtype=bool
        )
        indices[4] = 4
        return SampledPath(indices, mask, mask.sum(axis=1), 4)

    def test_shuffled_path_preserves_counts_and_validity(self) -> None:
        path = self.make_path()
        shuffled = shuffle_sampled_endpoints(path, 123)
        np.testing.assert_array_equal(path.mask, shuffled.mask)
        np.testing.assert_array_equal(
            np.sort(path.indices[path.mask]),
            np.sort(shuffled.indices[shuffled.mask]),
        )
        for center in range(len(path.indices)):
            neighbors = shuffled.indices[center, shuffled.mask[center]].tolist()
            self.assertNotIn(center, neighbors)
            self.assertEqual(len(neighbors), len(set(neighbors)))

    def test_no_neighbor_falls_back_to_self(self) -> None:
        torch.manual_seed(42)
        model = MetaPathRecommender(2, 5, {"actor": self.make_path()}, 8)
        vectors, weights = model.encode_movies()
        self.assertEqual(tuple(vectors.shape), (5, 8))
        self.assertEqual(tuple(weights.shape), (5, 2))
        self.assertAlmostEqual(float(weights[4, 0].detach()), 1.0)
        self.assertAlmostEqual(float(weights[4, 1].detach()), 0.0)
        self.assertTrue(bool(torch.isfinite(vectors).all()))

    def test_one_bpr_update_reaches_movie_encoder(self) -> None:
        torch.manual_seed(42)
        model = MetaPathRecommender(2, 5, {"actor": self.make_path()}, 8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        users = torch.tensor([0, 1])
        positives = torch.tensor([0, 2])
        negatives = torch.tensor([3, 4])
        before = model.movie_embedding.weight.detach().clone()
        movies, _ = model.encode_movies()
        loss = F.softplus(
            model.score_pairs(users, negatives, movies)
            - model.score_pairs(users, positives, movies)
        ).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.movie_embedding.weight))
        self.assertTrue(bool(torch.isfinite(model.movie_embedding.weight).all()))


if __name__ == "__main__":
    unittest.main()
