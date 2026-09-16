"""Run a deterministic user-neighborhood collaborative-filtering baseline.

Usage: python run_user_cf_baseline.py /path/to/hetrec2011-movielens-2k-v2
Only aggregate statistics are printed. Neighbor count is selected on validation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path

import pandas as pd

from data_protocol import load_split
from run_popularity_baseline import K, POSITIVE_RATING, evaluate


NEIGHBOR_CHOICES = (20, 50, 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--exclude-train-year-contradictions", action="store_true")
    args = parser.parse_args()
    train, validation, test, quality_audit = load_split(
        args.data_dir, args.exclude_train_year_contradictions
    )

    train_positive = train[train["rating"] >= POSITIVE_RATING]
    user_movies = train_positive.groupby("userID")["movieID"].agg(set).to_dict()
    movie_users = train_positive.groupby("movieID")["userID"].agg(set).to_dict()
    train_positive_users = set(user_movies)
    popularity = train_positive["movieID"].value_counts().to_dict()
    catalog = sorted(popularity, key=lambda movie_id: (-popularity[movie_id], movie_id))

    @lru_cache(maxsize=None)
    def sorted_neighbors(user_id: int) -> tuple[tuple[int, float], ...]:
        """Cosine similarity of positive train movie sets, excluding self."""
        overlap: Counter[int] = Counter()
        own_movies = user_movies[user_id]
        for movie_id in own_movies:
            overlap.update(movie_users[movie_id])
        overlap.pop(user_id, None)
        neighbors = [
            (
                other_user_id,
                shared / math.sqrt(len(own_movies) * len(user_movies[other_user_id])),
            )
            for other_user_id, shared in overlap.items()
        ]
        return tuple(sorted(neighbors, key=lambda pair: (-pair[1], pair[0])))

    def make_recommender(neighbor_count: int):
        def recommend(user_id: int, seen: set[int]) -> list[int]:
            score: dict[int, float] = defaultdict(float)
            for other_user_id, similarity in sorted_neighbors(user_id)[:neighbor_count]:
                for movie_id in user_movies[other_user_id]:
                    score[movie_id] += similarity

            # Positive neighbor votes rank first. Global popularity breaks ties and
            # fills the list when fewer than K movies received a neighbor vote.
            voted = sorted(
                (movie_id for movie_id in score if movie_id not in seen),
                key=lambda movie_id: (-score[movie_id], -popularity[movie_id], movie_id),
            )[:K]
            if len(voted) == K:
                return voted
            chosen = set(voted)
            fallback = [
                movie_id
                for movie_id in catalog
                if movie_id not in seen and movie_id not in chosen
            ][: K - len(voted)]
            return voted + fallback

        return recommend

    validation_by_neighbors = {
        str(neighbor_count): evaluate(
            validation,
            train,
            train_positive_users,
            catalog,
            recommend=make_recommender(neighbor_count),
        )
        for neighbor_count in NEIGHBOR_CHOICES
    }
    selected = max(
        NEIGHBOR_CHOICES,
        key=lambda count: (
            validation_by_neighbors[str(count)]["macro_ndcg_at_20"],
            validation_by_neighbors[str(count)]["macro_recall_at_20"],
            -count,
        ),
    )
    test_result = evaluate(
        test,
        pd.concat([train, validation]),
        train_positive_users,
        catalog,
        recommend=make_recommender(selected),
    )
    result = {
        "method": "positive-only user-user cosine KNN; similarity-weighted movie votes",
        "protocol": "same warm cohort, full train-positive catalog, seen-item exclusion and user-macro Top-20 metrics as popularity baseline",
        "neighbor_choices": list(NEIGHBOR_CHOICES),
        "selection": "highest validation macro NDCG@20; then recall; then smaller K",
        "selected_neighbors": selected,
        "validation_by_neighbors": validation_by_neighbors,
        "test": test_result,
    }
    if args.exclude_train_year_contradictions:
        result["data_quality_audit"] = quality_audit
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
