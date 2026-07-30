"""Tests for composing archived and current-season training rows."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_predictions.data.training import (
    TrainingTableError,
    load_training_tables,
)


def test_training_tables_union_columns_and_record_sources(tmp_path) -> None:
    historical_path = tmp_path / "historical.parquet"
    current_path = tmp_path / "current.parquet"
    pd.DataFrame(
        [
            {
                "season": "2024-25",
                "snapshot_gameweek": 1,
                "player_id": 1,
                "price": 5.0,
            }
        ]
    ).to_parquet(historical_path, index=False)
    pd.DataFrame(
        [
            {
                "season": "2026-27",
                "snapshot_gameweek": 1,
                "player_id": 2,
                "price": 6.0,
                "status": "a",
            }
        ]
    ).to_parquet(current_path, index=False)

    combined = load_training_tables([historical_path, current_path])

    assert len(combined) == 2
    assert "status" in combined
    assert combined["training_source_file"].nunique() == 2


def test_overlapping_training_sources_are_rejected(tmp_path) -> None:
    row = pd.DataFrame(
        [
            {
                "season": "2024-25",
                "snapshot_gameweek": 1,
                "player_id": 1,
            }
        ]
    )
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    row.to_parquet(first, index=False)
    row.to_parquet(second, index=False)

    with pytest.raises(TrainingTableError, match="overlap"):
        load_training_tables([first, second])
