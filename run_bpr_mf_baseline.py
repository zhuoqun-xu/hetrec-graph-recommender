"""Train and evaluate a positive-feedback BPR matrix-factorization baseline.

Usage: python run_bpr_mf_baseline.py /path/to/hetrec2011-movielens-2k-v2
Only aggregate metrics are printed. The original ratings are never exported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from data_protocol import load_split
from run_popularity_baseline import K, POSITIVE_RATING, evaluate


DIMENSIONS = 32
LEARNING_RATE = 0.02
REGULARIZATION = 0.005
BIAS_REGULARIZATION = 0.005
BATCH_SIZE = 1024
EPOCH_CHOICES = (5, 10, 15)
SEEDS = (42, 43, 44)


class BPRMF:
    def __init__(self, user_count: int, item_count: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.user_factors = self.rng.normal(
            0, 0.1, size=(user_count, DIMENSIONS)
        ).astype(np.float32)
        self.item_factors = self.rng.normal(
            0, 0.1, size=(item_count, DIMENSIONS)
        ).astype(np.float32)
        self.item_bias = np.zeros(item_count, dtype=np.float32)

    def train_epoch(
        self,
        positive_users: np.ndarray,
        positive_items: np.ndarray,
        seen_train: np.ndarray,
    ) -> float:
        ordering = self.rng.permutation(len(positive_users))
        loss_sum = 0.0
        for start in range(0, len(ordering), BATCH_SIZE):
            chosen = ordering[start : start + BATCH_SIZE]
            users = positive_users[chosen]
            items = positive_items[chosen]
            negatives = self.rng.integers(
                self.item_factors.shape[0], size=len(chosen), dtype=np.int32
            )
            invalid = seen_train[users, negatives]
            while invalid.any():
                negatives[invalid] = self.rng.integers(
                    self.item_factors.shape[0], size=int(invalid.sum()), dtype=np.int32
                )
                invalid = seen_train[users, negatives]

            user_before = self.user_factors[users].copy()
            positive_before = self.item_factors[items].copy()
            negative_before = self.item_factors[negatives].copy()
            positive_bias_before = self.item_bias[items].copy()
            negative_bias_before = self.item_bias[negatives].copy()

            difference = (
                np.sum(user_before * (positive_before - negative_before), axis=1)
                + positive_bias_before
                - negative_bias_before
            )
            loss_sum += float(np.logaddexp(0.0, -difference).sum())
            # Derivative of log(sigmoid(difference)) with respect to difference.
            weight = np.exp(-np.logaddexp(0.0, difference)).astype(np.float32)
            weight_column = weight[:, None]

            np.add.at(
                self.user_factors,
                users,
                LEARNING_RATE
                * (
                    weight_column * (positive_before - negative_before)
                    - REGULARIZATION * user_before
                ),
            )
            np.add.at(
                self.item_factors,
                items,
                LEARNING_RATE
                * (weight_column * user_before - REGULARIZATION * positive_before),
            )
            np.add.at(
                self.item_factors,
                negatives,
                LEARNING_RATE
                * (-weight_column * user_before - REGULARIZATION * negative_before),
            )
            np.add.at(
                self.item_bias,
                items,
                LEARNING_RATE
                * (weight - BIAS_REGULARIZATION * positive_bias_before),
            )
            np.add.at(
                self.item_bias,
                negatives,
                LEARNING_RATE
                * (-weight - BIAS_REGULARIZATION * negative_bias_before),
            )
        return loss_sum / len(positive_users)

    def recommend(self, user_index: int, seen_item_indices: np.ndarray, movie_ids: np.ndarray) -> list[int]:
        scores = self.item_factors @ self.user_factors[user_index] + self.item_bias
        scores[seen_item_indices] = -np.inf
        ranking = np.lexsort((movie_ids, -scores))[:K]
        return movie_ids[ranking].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--exclude-train-year-contradictions", action="store_true")
    args = parser.parse_args()
    train, validation, test, quality_audit = load_split(
        args.data_dir, args.exclude_train_year_contradictions
    )
    train_positive = train[train["rating"] >= POSITIVE_RATING]

    popularity = train_positive["movieID"].value_counts().to_dict()
    catalog = sorted(popularity, key=lambda movie_id: (-popularity[movie_id], movie_id))
    movie_ids = np.asarray(catalog, dtype=np.int64)
    item_index = {movie_id: index for index, movie_id in enumerate(catalog)}
    user_ids = sorted(train_positive["userID"].unique().tolist())
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    train_positive_users = set(user_ids)

    positive_users = train_positive["userID"].map(user_index).to_numpy(dtype=np.int32)
    positive_items = train_positive["movieID"].map(item_index).to_numpy(dtype=np.int32)
    # Sample negatives only from movies this user had not rated by train time.
    seen_train = np.zeros((len(user_ids), len(catalog)), dtype=bool)
    rated_catalog = train[
        train["userID"].isin(train_positive_users)
        & train["movieID"].isin(item_index)
    ]
    seen_train[
        rated_catalog["userID"].map(user_index).to_numpy(dtype=np.int32),
        rated_catalog["movieID"].map(item_index).to_numpy(dtype=np.int32),
    ] = True
    if not np.all(seen_train[positive_users, positive_items]):
        raise ValueError("A training positive is missing from the seen-item mask")
    if np.any(seen_train.all(axis=1)):
        raise ValueError("A user has no eligible training negative movie")

    def evaluate_model(model: BPRMF, held_out: pd.DataFrame, earlier: pd.DataFrame):
        def recommend(user_id: int, seen: set[int]) -> list[int]:
            seen_indices = np.fromiter(
                (item_index[movie_id] for movie_id in seen if movie_id in item_index),
                dtype=np.int32,
            )
            return model.recommend(user_index[user_id], seen_indices, movie_ids)

        return evaluate(
            held_out,
            earlier,
            train_positive_users,
            catalog,
            recommend=recommend,
        )

    validation_by_seed_and_epoch: dict[str, dict[str, dict]] = {}
    test_by_seed: dict[str, dict] = {}
    checkpoints: dict[int, dict[int, BPRMF]] = {}
    training_loss_by_seed: dict[str, dict[str, float]] = {}
    started = time.monotonic()
    for seed in SEEDS:
        model = BPRMF(len(user_ids), len(catalog), seed)
        validation_by_seed_and_epoch[str(seed)] = {}
        checkpoints[seed] = {}
        training_loss_by_seed[str(seed)] = {}
        for epoch in range(1, max(EPOCH_CHOICES) + 1):
            mean_loss = model.train_epoch(positive_users, positive_items, seen_train)
            if epoch in EPOCH_CHOICES:
                validation_by_seed_and_epoch[str(seed)][str(epoch)] = evaluate_model(
                    model, validation, train
                )
                training_loss_by_seed[str(seed)][str(epoch)] = mean_loss
                snapshot = BPRMF(len(user_ids), len(catalog), seed)
                snapshot.user_factors = model.user_factors.copy()
                snapshot.item_factors = model.item_factors.copy()
                snapshot.item_bias = model.item_bias.copy()
                checkpoints[seed][epoch] = snapshot

    validation_mean_ndcg = {
        epoch: sum(
            validation_by_seed_and_epoch[str(seed)][str(epoch)]["macro_ndcg_at_20"]
            for seed in SEEDS
        ) / len(SEEDS)
        for epoch in EPOCH_CHOICES
    }
    selected_epoch = max(
        EPOCH_CHOICES, key=lambda epoch: (validation_mean_ndcg[epoch], -epoch)
    )
    for seed in SEEDS:
        test_by_seed[str(seed)] = evaluate_model(
            checkpoints[seed][selected_epoch], test, pd.concat([train, validation])
        )

    metric_names = ("macro_recall_at_20", "macro_ndcg_at_20")
    test_summary = {}
    for metric in metric_names:
        values = np.asarray([test_by_seed[str(seed)][metric] for seed in SEEDS])
        test_summary[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
        }

    result = {
        "method": "BPR matrix factorization with user and movie embeddings plus movie bias",
        "protocol": "same global chronological split, warm cohort, full train-positive catalog, seen-item exclusion and user-macro Top-20 metrics as previous baselines",
        "dimensions": DIMENSIONS,
        "learning_rate": LEARNING_RATE,
        "regularization": REGULARIZATION,
        "bias_regularization": BIAS_REGULARIZATION,
        "batch_size": BATCH_SIZE,
        "negative_sampling": "uniform among train-positive catalog movies not rated by the user in train",
        "seeds": list(SEEDS),
        "epoch_choices": list(EPOCH_CHOICES),
        "selection": "highest mean validation macro NDCG@20 across seeds; shorter training on a tie",
        "validation_mean_ndcg_by_epoch": {str(k): v for k, v in validation_mean_ndcg.items()},
        "selected_epoch": selected_epoch,
        "validation_by_seed_and_epoch": validation_by_seed_and_epoch,
        "training_loss_by_seed_and_epoch": training_loss_by_seed,
        "test_by_seed": test_by_seed,
        "test_summary": test_summary,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    if args.exclude_train_year_contradictions:
        result["data_quality_audit"] = quality_audit
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
