"""Shared chronology and optional train-only movie-year sanity filter.

The original 80/10/10 cut points are selected before any exclusion. The
conservative sensitivity filter discovers contradictory movie IDs from train
only, then removes those movie IDs from all three partitions. It does not use
validation/test labels, timestamps, or outcomes to choose exclusions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_split(
    data_dir: Path, exclude_train_year_contradictions: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    ratings_path = data_dir / "user_ratedmovies-timestamps.dat"
    if not ratings_path.is_file():
        raise FileNotFoundError(f"Missing source file: {ratings_path}")
    ratings = pd.read_csv(ratings_path, sep="\t")
    if ratings.duplicated(["userID", "movieID"]).any():
        raise ValueError("Duplicate user/movie ratings need an explicit policy")
    ratings = ratings.sort_values("timestamp", kind="stable").reset_index(drop=True)
    first_cut = int(len(ratings) * 0.8)
    second_cut = int(len(ratings) * 0.9)
    train = ratings.iloc[:first_cut]
    validation = ratings.iloc[first_cut:second_cut]
    test = ratings.iloc[second_cut:]
    audit: dict[str, int] = {
        "original_train_ratings": len(train),
        "original_validation_ratings": len(validation),
        "original_test_ratings": len(test),
    }
    if not exclude_train_year_contradictions:
        return train, validation, test, audit

    movies_path = data_dir / "movies.dat"
    if not movies_path.is_file():
        raise FileNotFoundError(f"Missing source file: {movies_path}")
    movie_year = (
        pd.read_csv(
            movies_path, sep="\t", usecols=["id", "year"], encoding="latin1"
        )
        .set_index("id")["year"]
        .pipe(pd.to_numeric, errors="coerce")
    )
    rating_year_utc = pd.to_datetime(train["timestamp"], unit="ms", utc=True).dt.year
    listed_year = train["movieID"].map(movie_year)
    contradictory_movies = set(
        train.loc[listed_year > rating_year_utc, "movieID"].tolist()
    )
    if not contradictory_movies:
        raise ValueError("Expected year contradictions were not found in training")

    original_lengths = (len(train), len(validation), len(test))
    train, validation, test = (
        frame.loc[~frame["movieID"].isin(contradictory_movies)].copy()
        for frame in (train, validation, test)
    )
    audit.update(
        {
            "excluded_movie_ids_found_from_train_only": len(contradictory_movies),
            "removed_train_ratings": original_lengths[0] - len(train),
            "removed_validation_ratings": original_lengths[1] - len(validation),
            "removed_test_ratings": original_lengths[2] - len(test),
            "remaining_train_ratings": len(train),
            "remaining_validation_ratings": len(validation),
            "remaining_test_ratings": len(test),
        }
    )
    return train, validation, test, audit
