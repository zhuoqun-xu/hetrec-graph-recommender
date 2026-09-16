"""Run a reproducible warm-user/warm-item Top-20 popularity baseline.

Usage: python run_popularity_baseline.py /path/to/hetrec2011-movielens-2k-v2
Only aggregate statistics are printed. Source ratings are never exported.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import pandas as pd

from data_protocol import load_split


K = 20
POSITIVE_RATING = 4.0


def evaluate(
    held_out: pd.DataFrame,
    earlier_ratings: pd.DataFrame,
    train_positive_users: set[int],
    catalog: list[int],
    recommend: Callable[[int, set[int]], list[int]] | None = None,
) -> dict[str, int | float]:
    catalog_set = set(catalog)
    relevant = held_out[
        (held_out["rating"] >= POSITIVE_RATING)
        & held_out["userID"].isin(train_positive_users)
        & held_out["movieID"].isin(catalog_set)
    ]
    truth_by_user = relevant.groupby("userID")["movieID"].agg(set).to_dict()
    seen_by_user = earlier_ratings.groupby("userID")["movieID"].agg(set).to_dict()

    recall_values: list[float] = []
    ndcg_values: list[float] = []
    hit_count = 0
    recommended_count = 0
    for user_id, truth in truth_by_user.items():
        seen = seen_by_user.get(user_id, set())
        recommendations = (
            recommend(user_id, seen)
            if recommend is not None
            else [movie_id for movie_id in catalog if movie_id not in seen][:K]
        )
        if (
            len(recommendations) > K
            or len(recommendations) != len(set(recommendations))
            or any(movie_id in seen or movie_id not in catalog_set for movie_id in recommendations)
        ):
            raise ValueError(f"Invalid recommendations for user {user_id}")
        recommended_count += len(recommendations)
        hits = sum(movie_id in truth for movie_id in recommendations)
        hit_count += hits
        recall_values.append(hits / len(truth))

        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, movie_id in enumerate(recommendations)
            if movie_id in truth
        )
        ideal_dcg = sum(
            1.0 / math.log2(rank + 2) for rank in range(min(K, len(truth)))
        )
        ndcg_values.append(dcg / ideal_dcg)

    if not truth_by_user:
        raise ValueError("No eligible users have warm held-out positives")

    return {
        "eligible_users": len(truth_by_user),
        "eligible_positive_ratings": len(relevant),
        "candidate_movies_before_exclusion": len(catalog),
        "recommended_movies_total": recommended_count,
        "top20_hits_total": hit_count,
        "macro_recall_at_20": sum(recall_values) / len(recall_values),
        "macro_ndcg_at_20": sum(ndcg_values) / len(ndcg_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--exclude-train-year-contradictions", action="store_true")
    args = parser.parse_args()
    train, validation, test, quality_audit = load_split(
        args.data_dir, args.exclude_train_year_contradictions
    )

    train_positive = train[train["rating"] >= POSITIVE_RATING]
    train_positive_users = set(train_positive["userID"])
    popularity = train_positive["movieID"].value_counts().to_dict()
    # A movie's score is its train-period positive count; ties use movie ID.
    catalog = sorted(popularity, key=lambda movie_id: (-popularity[movie_id], movie_id))

    result = {
        "protocol": {
            "split": "global timestamp order, stable ties, 80/10/10 by rating count",
            "positive_rating_at_least": POSITIVE_RATING,
            "ranking": "training positive count descending; movie ID ascending on ties",
            "k": K,
            "cohort": "users and movies with at least one positive training rating",
            "candidate_set": "all training-positive movies, excluding all earlier rated movies per user",
            "validation_seen": "all train ratings",
            "test_seen": "all train and validation ratings",
            "recall": "hits in top K / held-out warm positive movies, averaged by user",
            "ndcg": "binary DCG / ideal binary DCG, averaged by user",
        },
        "split_counts": {
            "train_ratings": len(train),
            "validation_ratings": len(validation),
            "test_ratings": len(test),
            "train_positive_ratings": len(train_positive),
        },
        "validation": evaluate(validation, train, train_positive_users, catalog),
        "test": evaluate(
            test,
            pd.concat([train, validation]),
            train_positive_users,
            catalog,
        ),
    }
    if args.exclude_train_year_contradictions:
        result["protocol"]["data_quality_rule"] = (
            "original cut points first; exclude every movie whose listed year "
            "is after any train rating year; apply IDs to all partitions"
        )
        result["data_quality_audit"] = quality_audit
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
