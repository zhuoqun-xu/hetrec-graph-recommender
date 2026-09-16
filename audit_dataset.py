"""Read-only audit of the official HetRec 2011 MovieLens-2k v2 files.

Usage: python audit_dataset.py /path/to/extracted/hetrec2011-movielens-2k-v2
The script prints aggregate statistics only; it does not export source data.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def utc_time(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    data_dir = args.data_dir

    required = {
        "ratings": "user_ratedmovies-timestamps.dat",
        "movies": "movies.dat",
        "actors": "movie_actors.dat",
        "directors": "movie_directors.dat",
        "genres": "movie_genres.dat",
    }
    missing = [name for name in required.values() if not (data_dir / name).is_file()]
    if missing:
        parser.error(f"Missing required files: {missing}")

    ratings = pd.read_csv(data_dir / required["ratings"], sep="\t")
    movies = pd.read_csv(
        data_dir / required["movies"], sep="\t", usecols=["id"], encoding="latin1"
    )
    actors = pd.read_csv(data_dir / required["actors"], sep="\t", encoding="latin1")
    directors = pd.read_csv(
        data_dir / required["directors"], sep="\t", encoding="latin1"
    )
    genres = pd.read_csv(data_dir / required["genres"], sep="\t", encoding="latin1")

    movie_ids = set(movies["id"])
    rated_movie_ids = set(ratings["movieID"])
    relation_ids = {
        "actors": set(actors["movieID"]),
        "directors": set(directors["movieID"]),
        "genres": set(genres["movieID"]),
    }

    ratings = ratings.sort_values("timestamp", kind="stable").reset_index(drop=True)
    first_cut = int(len(ratings) * 0.8)
    second_cut = int(len(ratings) * 0.9)
    train = ratings.iloc[:first_cut]
    validation = ratings.iloc[first_cut:second_cut]
    test = ratings.iloc[second_cut:]

    train_positive = train[train["rating"] >= 4]
    test_positive = test[test["rating"] >= 4]
    train_users = set(train_positive["userID"])
    train_items = set(train_positive["movieID"])
    known_user_test = test_positive[test_positive["userID"].isin(train_users)]
    new_user_test = test_positive[~test_positive["userID"].isin(train_users)]
    warm_test = test_positive[
        test_positive["userID"].isin(train_users)
        & test_positive["movieID"].isin(train_items)
    ]
    cold_item_test = test_positive[
        test_positive["userID"].isin(train_users)
        & ~test_positive["movieID"].isin(train_items)
    ]

    top_five_actors = (
        actors.sort_values(["movieID", "ranking", "actorID"], kind="stable")
        .groupby("movieID", sort=False)
        .head(5)
    )

    result = {
        "source_counts": {
            "ratings": len(ratings),
            "users": ratings["userID"].nunique(),
            "rated_movies": ratings["movieID"].nunique(),
            "all_movies": len(movies),
            "actor_edges": len(actors),
            "unique_actors": actors["actorID"].nunique(),
            "director_edges": len(directors),
            "unique_directors": directors["directorID"].nunique(),
            "genre_edges": len(genres),
            "unique_genres": genres["genre"].nunique(),
        },
        "joins": {
            "rating_movie_ids_missing_from_movies": len(rated_movie_ids - movie_ids),
            "actor_movie_ids_missing_from_movies": len(relation_ids["actors"] - movie_ids),
            "director_movie_ids_missing_from_movies": len(relation_ids["directors"] - movie_ids),
            "genre_movie_ids_missing_from_movies": len(relation_ids["genres"] - movie_ids),
            "rated_movies_without_actor": len(rated_movie_ids - relation_ids["actors"]),
            "rated_movies_without_director": len(rated_movie_ids - relation_ids["directors"]),
            "rated_movies_without_genre": len(rated_movie_ids - relation_ids["genres"]),
            "duplicate_user_movie_ratings": int(
                ratings.duplicated(["userID", "movieID"]).sum()
            ),
        },
        "ratings": {
            "min": float(ratings["rating"].min()),
            "max": float(ratings["rating"].max()),
            "counts_by_value": {
                str(value): int(count)
                for value, count in ratings["rating"].value_counts().sort_index().items()
            },
            "positive_ge_4": int((ratings["rating"] >= 4).sum()),
            "timestamp_min_utc": utc_time(int(ratings["timestamp"].min())),
            "timestamp_max_utc": utc_time(int(ratings["timestamp"].max())),
        },
        "top_five_actor_projection": {
            "actor_edges": len(top_five_actors),
            "unique_actors": top_five_actors["actorID"].nunique(),
            "covered_movies": top_five_actors["movieID"].nunique(),
        },
        "provisional_global_80_10_10_split": {
            "train_ratings": len(train),
            "validation_ratings": len(validation),
            "test_ratings": len(test),
            "train_last_timestamp_utc": utc_time(int(train["timestamp"].max())),
            "validation_first_timestamp_utc": utc_time(
                int(validation["timestamp"].min())
            ),
            "validation_last_timestamp_utc": utc_time(
                int(validation["timestamp"].max())
            ),
            "test_first_timestamp_utc": utc_time(int(test["timestamp"].min())),
            "train_positive_edges": len(train_positive),
            "train_positive_users": len(train_users),
            "train_positive_movies": len(train_items),
            "test_positive_edges": len(test_positive),
            "test_positive_users": test_positive["userID"].nunique(),
            "known_user_test_positive_edges": len(known_user_test),
            "known_user_test_users": known_user_test["userID"].nunique(),
            "new_user_test_positive_edges": len(new_user_test),
            "new_user_test_users": new_user_test["userID"].nunique(),
            "warm_test_positive_edges": len(warm_test),
            "warm_test_users": warm_test["userID"].nunique(),
            "cold_item_test_positive_edges": len(cold_item_test),
            "cold_item_test_movies": cold_item_test["movieID"].nunique(),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
