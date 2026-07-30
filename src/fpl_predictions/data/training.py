"""Composition and validation of historical and current training tables."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


class TrainingTableError(ValueError):
    """Raised when multiple training sources cannot be combined safely."""


def load_training_tables(paths: Sequence[Path]) -> pd.DataFrame:
    """Union compatible training tables while preserving source provenance."""
    if not paths:
        raise TrainingTableError("At least one training table path is required")
    frames = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Training table not found: {path}")
        frame = pd.read_parquet(path)
        frame = frame.copy()
        frame["training_source_file"] = str(path.resolve())
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    required = {"season", "snapshot_gameweek", "player_id"}
    missing = required.difference(combined.columns)
    if missing:
        raise TrainingTableError(
            "Training tables are missing identity columns: "
            + ", ".join(sorted(missing))
        )
    key = ["season", "snapshot_gameweek", "player_id"]
    duplicated = combined.duplicated(key, keep=False)
    if duplicated.any():
        examples = (
            combined.loc[duplicated, key]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise TrainingTableError(
            "Training sources overlap on season/gameweek/player keys: "
            f"{examples}"
        )
    return combined.sort_values(key, ignore_index=True)

