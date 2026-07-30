"""Tests for purged rolling-origin validation."""

from __future__ import annotations

import pandas as pd

from fpl_predictions.modelling.validation import rolling_origin_folds


def test_same_season_training_labels_finish_before_validation() -> None:
    frame = pd.DataFrame(
        [
            {
                "season": "2026-27",
                "snapshot_gameweek": gameweek,
                "snapshot_timestamp": f"2026-09-{gameweek:02d}T10:00:00Z",
            }
            for gameweek in range(1, 11)
        ]
    )

    folds = rolling_origin_folds(frame, horizon=3, min_train_periods=2)
    fold = next(item for item in folds if item.validation_gameweek == 8)
    training_gameweeks = frame.loc[
        fold.train_index, "snapshot_gameweek"
    ].tolist()

    assert max(training_gameweeks) == 5
    assert frame.loc[
        fold.validation_index, "snapshot_gameweek"
    ].tolist() == [8]


def test_prior_season_is_available_without_same_season_gameweek_math() -> None:
    frame = pd.DataFrame(
        [
            {
                "season": "2025-26",
                "snapshot_gameweek": 38,
                "snapshot_timestamp": "2026-05-20T10:00:00Z",
            },
            {
                "season": "2026-27",
                "snapshot_gameweek": 1,
                "snapshot_timestamp": "2026-08-15T10:00:00Z",
            },
        ]
    )

    folds = rolling_origin_folds(frame, horizon=5, min_train_periods=1)

    assert folds[0].validation_gameweek == 1
    assert frame.loc[folds[0].train_index, "season"].tolist() == ["2025-26"]

