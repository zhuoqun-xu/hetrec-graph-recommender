"""Unit tests for chronological splitting and train-only filtering."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from data_protocol import load_split


class DataProtocolTests(unittest.TestCase):
    def write_fixture(self, directory: Path) -> None:
        timestamps = pd.to_datetime(
            [
                "2019-01-01",
                "2019-02-01",
                "2019-03-01",
                "2019-04-01",
                "2019-05-01",
                "2019-06-01",
                "2019-07-01",
                "2019-08-01",
                "2021-01-01",
                "2021-02-01",
            ],
            utc=True,
        )
        ratings = pd.DataFrame(
            {
                "userID": range(1, 11),
                "movieID": [99, 1, 2, 3, 4, 5, 6, 7, 99, 99],
                "rating": [4.0] * 10,
                "timestamp": (timestamps.astype("int64") // 1_000_000).tolist(),
            }
        )
        ratings.to_csv(
            directory / "user_ratedmovies-timestamps.dat", sep="\t", index=False
        )
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7, 99],
                "year": [2010, 2010, 2010, 2010, 2010, 2010, 2010, 2020],
            }
        ).to_csv(directory / "movies.dat", sep="\t", index=False)

    def test_cut_points_precede_train_only_filter(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_fixture(directory)
            train, validation, test, audit = load_split(
                directory, exclude_train_year_contradictions=True
            )

        self.assertEqual(audit["original_train_ratings"], 8)
        self.assertEqual(audit["original_validation_ratings"], 1)
        self.assertEqual(audit["original_test_ratings"], 1)
        self.assertEqual(audit["excluded_movie_ids_found_from_train_only"], 1)
        self.assertEqual((len(train), len(validation), len(test)), (7, 0, 0))
        self.assertNotIn(99, train["movieID"].tolist())

    def test_unfiltered_split_retains_original_rows(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_fixture(directory)
            train, validation, test, audit = load_split(directory)

        self.assertEqual((len(train), len(validation), len(test)), (8, 1, 1))
        self.assertNotIn("excluded_movie_ids_found_from_train_only", audit)


if __name__ == "__main__":
    unittest.main()
