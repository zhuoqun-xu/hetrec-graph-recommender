"""Movie meta-path indexing and reproducible neighbor sampling.

Only train-positive candidate movie IDs are admitted as source or destination
movie nodes. This avoids silently bringing post-cutoff or cold movie IDs into
the warm-item encoder. No complete genre movie-pair projection is stored.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PATH_COLUMNS = {
    "actor": ("movie_actors.dat", "actorID"),
    "director": ("movie_directors.dat", "directorID"),
    "genre": ("movie_genres.dat", "genre"),
}


@dataclass
class SampledPath:
    indices: np.ndarray  # item_count x sample_count, padded with self index
    mask: np.ndarray  # true only for sampled, non-self meta-path neighbors
    potential_neighbor_counts: np.ndarray
    metadata_covered_movies: int

    def summary(self) -> dict[str, int | float]:
        counts = self.potential_neighbor_counts
        return {
            "candidate_movies": len(counts),
            "metadata_covered_movies": self.metadata_covered_movies,
            "movies_with_metapath_neighbors": int((counts > 0).sum()),
            "movies_without_metapath_neighbors": int((counts == 0).sum()),
            "median_potential_neighbors": float(np.median(counts)),
            "p90_potential_neighbors": float(np.percentile(counts, 90)),
            "max_potential_neighbors": int(counts.max()),
            "sampled_directed_edges": int(self.mask.sum()),
        }


def load_path_index(
    data_dir: Path, catalog: list[int], path_name: str
) -> tuple[dict[int, set], dict[object, set[int]]]:
    if path_name not in PATH_COLUMNS:
        raise ValueError(f"Unknown path: {path_name}")
    filename, attribute_column = PATH_COLUMNS[path_name]
    source = data_dir / filename
    if not source.is_file():
        raise FileNotFoundError(f"Missing source file: {source}")
    frame = pd.read_csv(source, sep="\t", encoding="latin1")
    if path_name == "actor":
        # Keep a movie's five earliest-ranked actors before catalog filtering.
        frame = (
            frame.sort_values(["movieID", "ranking", "actorID"], kind="stable")
            .groupby("movieID", sort=False)
            .head(5)
        )
    item_index = {movie_id: index for index, movie_id in enumerate(catalog)}
    frame = frame[frame["movieID"].isin(item_index)]

    movie_attributes: dict[int, set] = defaultdict(set)
    attribute_movies: dict[object, set[int]] = defaultdict(set)
    for movie_id, attribute in frame[["movieID", attribute_column]].itertuples(
        index=False, name=None
    ):
        item_id = item_index[movie_id]
        movie_attributes[item_id].add(attribute)
        attribute_movies[attribute].add(item_id)
    return dict(movie_attributes), dict(attribute_movies)


def sample_path(
    data_dir: Path,
    catalog: list[int],
    path_name: str,
    sample_count: int,
    seed: int,
) -> SampledPath:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    movie_attributes, attribute_movies = load_path_index(
        data_dir, catalog, path_name
    )
    rng = np.random.default_rng(seed)
    item_count = len(catalog)
    indices = np.broadcast_to(
        np.arange(item_count, dtype=np.int32)[:, None],
        (item_count, sample_count),
    ).copy()
    mask = np.zeros((item_count, sample_count), dtype=bool)
    potential_counts = np.zeros(item_count, dtype=np.int32)
    for item_id in range(item_count):
        candidates: set[int] = set()
        for attribute in movie_attributes.get(item_id, ()):
            candidates.update(attribute_movies[attribute])
        candidates.discard(item_id)
        potential_counts[item_id] = len(candidates)
        if not candidates:
            continue
        candidates_sorted = np.asarray(sorted(candidates), dtype=np.int32)
        chosen = (
            rng.choice(candidates_sorted, size=sample_count, replace=False)
            if len(candidates_sorted) > sample_count
            else candidates_sorted
        )
        indices[item_id, : len(chosen)] = chosen
        mask[item_id, : len(chosen)] = True
    return SampledPath(
        indices,
        mask,
        potential_counts,
        metadata_covered_movies=len(movie_attributes),
    )


def shuffle_sampled_endpoints(sampled: SampledPath, seed: int) -> SampledPath:
    """Null control: keep row degree/endpoints, but forbid self/duplicate edges."""
    rng = np.random.default_rng(seed)
    indices = sampled.indices.copy()
    endpoints = indices[sampled.mask].copy()
    rng.shuffle(endpoints)
    indices[sampled.mask] = endpoints
    rows, columns = np.where(sampled.mask)
    for position in range(len(rows)):
        row, column = int(rows[position]), int(columns[position])
        destination = int(indices[row, column])
        other_destinations = indices[row, sampled.mask[row]].tolist()
        other_destinations.remove(destination)
        if destination != row and destination not in other_destinations:
            continue
        for _ in range(1000):
            partner = int(rng.integers(len(rows)))
            partner_row, partner_column = int(rows[partner]), int(columns[partner])
            if partner_row == row:
                continue
            partner_destination = int(indices[partner_row, partner_column])
            partner_others = indices[
                partner_row, sampled.mask[partner_row]
            ].tolist()
            partner_others.remove(partner_destination)
            if (
                partner_destination == row
                or partner_destination in other_destinations
                or destination == partner_row
                or destination in partner_others
            ):
                continue
            indices[row, column] = partner_destination
            indices[partner_row, partner_column] = destination
            break
        else:
            raise RuntimeError("Could not repair a shuffled meta-path edge")
    return SampledPath(
        indices,
        sampled.mask.copy(),
        sampled.potential_neighbor_counts.copy(),
        sampled.metadata_covered_movies,
    )
