"""Train a thesis-inspired warm-item movie meta-path recommender.

Usage: python -B run_meta_path_recommender.py /path/to/hetrec2011-movielens-2k-v2
The raw licensed data is read locally and only aggregate results are printed.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from data_protocol import load_split
from graph_data import sample_path, shuffle_sampled_endpoints
from meta_path_model import MetaPathRecommender
from run_popularity_baseline import K, POSITIVE_RATING, evaluate


DIMENSIONS = 32
SAMPLES_PER_PATH = 10
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.001
EPOCH_CHOICES = (80, 160, 240)
SEEDS = (42, 43, 44)
PATHS = ("actor", "director")


def sample_negatives(
    rng: np.random.Generator,
    positive_users: np.ndarray,
    seen_train: np.ndarray,
) -> np.ndarray:
    negatives = rng.integers(
        seen_train.shape[1], size=len(positive_users), dtype=np.int32
    )
    invalid = seen_train[positive_users, negatives]
    while invalid.any():
        negatives[invalid] = rng.integers(
            seen_train.shape[1], size=int(invalid.sum()), dtype=np.int32
        )
        invalid = seen_train[positive_users, negatives]
    return negatives


def build_interactions(train: pd.DataFrame) -> dict:
    train_positive = train[train["rating"] >= POSITIVE_RATING]
    popularity = train_positive["movieID"].value_counts().to_dict()
    catalog = sorted(popularity, key=lambda movie_id: (-popularity[movie_id], movie_id))
    movie_ids = np.asarray(catalog, dtype=np.int64)
    item_index = {movie_id: index for index, movie_id in enumerate(catalog)}
    user_ids = sorted(train_positive["userID"].unique().tolist())
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    positive_users = train_positive["userID"].map(user_index).to_numpy(dtype=np.int32)
    positive_items = train_positive["movieID"].map(item_index).to_numpy(dtype=np.int32)
    seen_train = np.zeros((len(user_ids), len(catalog)), dtype=bool)
    rated_catalog = train[
        train["userID"].isin(user_index) & train["movieID"].isin(item_index)
    ]
    seen_train[
        rated_catalog["userID"].map(user_index).to_numpy(dtype=np.int32),
        rated_catalog["movieID"].map(item_index).to_numpy(dtype=np.int32),
    ] = True
    if not np.all(seen_train[positive_users, positive_items]):
        raise ValueError("Training positive missing from seen mask")
    if np.any(seen_train.all(axis=1)):
        raise ValueError("Some training users have no unrated candidate")
    return {
        "catalog": catalog,
        "movie_ids": movie_ids,
        "item_index": item_index,
        "user_ids": user_ids,
        "user_index": user_index,
        "positive_users": positive_users,
        "positive_items": positive_items,
        "seen_train": seen_train,
    }


def evaluate_scores(
    scores: np.ndarray,
    held_out: pd.DataFrame,
    earlier: pd.DataFrame,
    interactions: dict,
) -> dict:
    catalog = interactions["catalog"]
    movie_ids = interactions["movie_ids"]
    item_index = interactions["item_index"]
    user_index = interactions["user_index"]

    def recommend(user_id: int, seen: set[int]) -> list[int]:
        row = scores[user_index[user_id]].copy()
        seen_indices = np.fromiter(
            (item_index[movie_id] for movie_id in seen if movie_id in item_index),
            dtype=np.int32,
        )
        row[seen_indices] = -np.inf
        order = np.lexsort((movie_ids, -row))
        return movie_ids[order[:K]].tolist()

    return evaluate(
        held_out,
        earlier,
        set(interactions["user_ids"]),
        catalog,
        recommend=recommend,
    )


def train_variant(
    seed: int,
    paths: dict,
    interactions: dict,
    variant: str,
    max_epoch: int,
    validation: pd.DataFrame | None = None,
    train: pd.DataFrame | None = None,
) -> tuple[MetaPathRecommender, dict, dict]:
    torch.manual_seed(seed)
    model = MetaPathRecommender(
        len(interactions["user_ids"]),
        len(interactions["catalog"]),
        paths,
        dimensions=DIMENSIONS,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    rng = np.random.default_rng(seed + 10000)
    users_numpy = interactions["positive_users"]
    users = torch.from_numpy(users_numpy.astype(np.int64))
    positives = torch.from_numpy(interactions["positive_items"].astype(np.int64))
    enabled = {
        "self_only": set(),
        "actor_only": {"actor"},
        "director_only": {"director"},
    }.get(variant, set(PATHS))
    equal_attention = variant == "equal_attention"
    validation_by_epoch: dict[str, dict] = {}
    loss_by_epoch: dict[str, float] = {}
    checkpoints: dict[int, dict] = {}
    for epoch in range(1, max_epoch + 1):
        negatives_numpy = sample_negatives(
            rng, users_numpy, interactions["seen_train"]
        )
        negatives = torch.from_numpy(negatives_numpy.astype(np.int64))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        movie_vectors, _ = model.encode_movies(
            enabled_paths=enabled, equal_attention=equal_attention
        )
        positive_scores = model.score_pairs(users, positives, movie_vectors)
        negative_scores = model.score_pairs(users, negatives, movie_vectors)
        loss = F.softplus(negative_scores - positive_scores).mean()
        loss.backward()
        optimizer.step()
        if epoch in EPOCH_CHOICES or epoch == max_epoch:
            loss_by_epoch[str(epoch)] = float(loss.detach())
            if validation is not None and train is not None:
                model.eval()
                scores, _ = model.full_score_matrix(
                    enabled_paths=enabled, equal_attention=equal_attention
                )
                validation_by_epoch[str(epoch)] = evaluate_scores(
                    scores, validation, train, interactions
                )
                checkpoints[epoch] = copy.deepcopy(model.state_dict())
            print(
                f"{variant} seed={seed} epoch={epoch} loss={loss_by_epoch[str(epoch)]:.4f}",
                file=sys.stderr,
                flush=True,
            )
    return model, validation_by_epoch, {
        "loss": loss_by_epoch,
        "checkpoints": checkpoints,
    }


def summarize(metrics_by_seed: dict[str, dict]) -> dict:
    summary: dict[str, dict[str, float]] = {}
    for metric in ("macro_recall_at_20", "macro_ndcg_at_20"):
        values = np.asarray(
            [metrics_by_seed[str(seed)][metric] for seed in SEEDS], dtype=float
        )
        summary[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--exclude-train-year-contradictions", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    train, validation, test, data_quality_audit = load_split(
        args.data_dir, args.exclude_train_year_contradictions
    )
    interactions = build_interactions(train)
    catalog = interactions["catalog"]
    paths_by_seed: dict[int, dict] = {}
    path_summary = {}
    for seed in SEEDS:
        paths_by_seed[seed] = {
            name: sample_path(
                args.data_dir, catalog, name, SAMPLES_PER_PATH, seed + 101 * offset
            )
            for offset, name in enumerate(PATHS)
        }
        if seed == SEEDS[0]:
            path_summary = {
                name: sampled.summary()
                for name, sampled in paths_by_seed[seed].items()
            }

    validation_by_seed_and_epoch: dict[str, dict] = {}
    loss_by_seed_and_epoch: dict[str, dict] = {}
    checkpoint_by_seed_and_epoch: dict[int, dict] = {}
    for seed in SEEDS:
        _, validation_results, training = train_variant(
            seed,
            paths_by_seed[seed],
            interactions,
            "real_graph",
            max(EPOCH_CHOICES),
            validation=validation,
            train=train,
        )
        validation_by_seed_and_epoch[str(seed)] = validation_results
        loss_by_seed_and_epoch[str(seed)] = training["loss"]
        checkpoint_by_seed_and_epoch[seed] = training["checkpoints"]

    validation_mean_ndcg = {
        epoch: float(
            np.mean(
                [
                    validation_by_seed_and_epoch[str(seed)][str(epoch)][
                        "macro_ndcg_at_20"
                    ]
                    for seed in SEEDS
                ]
            )
        )
        for epoch in EPOCH_CHOICES
    }
    selected_epoch = max(
        EPOCH_CHOICES, key=lambda epoch: (validation_mean_ndcg[epoch], -epoch)
    )
    earlier_test = pd.concat([train, validation])
    test_by_variant: dict[str, dict[str, dict]] = {
        "real_graph": {},
        "shuffled_graph": {},
        "self_only": {},
        "actor_only": {},
        "director_only": {},
        "equal_attention": {},
    }
    attention_summary: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        real_model = MetaPathRecommender(
            len(interactions["user_ids"]),
            len(catalog),
            paths_by_seed[seed],
            DIMENSIONS,
        )
        real_model.load_state_dict(
            checkpoint_by_seed_and_epoch[seed][selected_epoch]
        )
        real_model.eval()
        real_scores, weights = real_model.full_score_matrix()
        test_by_variant["real_graph"][str(seed)] = evaluate_scores(
            real_scores, test, earlier_test, interactions
        )
        attention_summary[str(seed)] = {
            name: float(weights[:, index].mean())
            for index, name in enumerate(("self", *PATHS))
        }

        shuffled_paths = {
            name: shuffle_sampled_endpoints(sampled, seed + 707 + offset)
            for offset, (name, sampled) in enumerate(paths_by_seed[seed].items())
        }
        for variant, variant_paths, enabled in (
            ("shuffled_graph", shuffled_paths, set(PATHS)),
            ("self_only", paths_by_seed[seed], set()),
            ("actor_only", paths_by_seed[seed], {"actor"}),
            ("director_only", paths_by_seed[seed], {"director"}),
            ("equal_attention", paths_by_seed[seed], set(PATHS)),
        ):
            model, _, _ = train_variant(
                seed,
                variant_paths,
                interactions,
                variant,
                selected_epoch,
            )
            model.eval()
            scores, _ = model.full_score_matrix(
                enabled_paths=enabled,
                equal_attention=variant == "equal_attention",
            )
            test_by_variant[variant][str(seed)] = evaluate_scores(
                scores, test, earlier_test, interactions
            )

    result = {
        "experiment_status": "exploratory extended schedule after the first 20/40/80 run showed rising validation performance; original test outcomes were already visible, so this is not a fresh blind test",
        "method": "warm-item movie meta-path GraphSAGE-style mean aggregation, movie-level semantic attention, user-item BPR ranking",
        "thesis_relation": "adaptation, not an exact reproduction or inductive/cold-start claim",
        "protocol": {
            "split": "original global chronological 80/10/10 cut points",
            "quality_filter": "training-detected movie-year contradictions removed in all partitions"
            if args.exclude_train_year_contradictions
            else "none; original baseline split",
            "cohort": "warm user and warm movie with at least one train positive",
            "candidate_set": "all train-positive movies, excluding all earlier ratings for each user",
            "validation_seen": "train ratings",
            "test_seen": "train and validation ratings",
            "k": K,
            "positive_rating_at_least": POSITIVE_RATING,
            "metrics": "user-macro binary Recall@20 and NDCG@20",
        },
        "training": {
            "dimensions": DIMENSIONS,
            "samples_per_path": SAMPLES_PER_PATH,
            "paths": list(PATHS),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW; one full-positive batch update per epoch",
            "determinism": "one CPU thread, deterministic PyTorch algorithms, fixed NumPy/PyTorch seeds",
            "negative_sampling": "one uniformly sampled train-unrated warm movie per train positive per epoch",
            "seeds": list(SEEDS),
            "epoch_choices": list(EPOCH_CHOICES),
            "selection": "highest mean validation macro NDCG@20 across three real-graph seeds; shorter on tie",
            "matched_controls": "same model dimensions, optimizer, init seed, negatives and selected epoch; shuffled endpoints preserve center degree and global endpoint multiset without self or duplicate edges; self-only/actor-only/director-only mask paths; equal-attention replaces learned attention",
        },
        "split_counts": {
            "train_ratings": len(train),
            "validation_ratings": len(validation),
            "test_ratings": len(test),
            "train_positive_ratings": len(interactions["positive_users"]),
            "train_positive_users": len(interactions["user_ids"]),
            "candidate_movies": len(catalog),
        },
        "path_summary_seed_42": path_summary,
        "validation_mean_ndcg_by_epoch": {
            str(epoch): value for epoch, value in validation_mean_ndcg.items()
        },
        "validation_by_seed_and_epoch": validation_by_seed_and_epoch,
        "training_loss_by_seed_and_epoch": loss_by_seed_and_epoch,
        "selected_epoch": selected_epoch,
        "test_by_variant": test_by_variant,
        "test_summary_by_variant": {
            variant: summarize(per_seed)
            for variant, per_seed in test_by_variant.items()
        },
        "mean_attention_weights_by_seed": attention_summary,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    if args.exclude_train_year_contradictions:
        result["data_quality_audit"] = data_quality_audit
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
