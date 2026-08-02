"""Tests for leakage-safe feature and label construction."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_predictions.data.features import (
    FeatureValidationError,
    build_prediction_table,
    build_training_table,
)


def _snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": "2026-27",
                "snapshot_gameweek": 4,
                "snapshot_timestamp": "2026-09-01T09:00:00Z",
                "snapshot_deadline_time": "2026-09-01T10:00:00Z",
                "player_id": 101,
                "club_id": 10,
                "price": 7.5,
            },
            {
                "season": "2026-27",
                "snapshot_gameweek": 4,
                "snapshot_timestamp": "2026-09-01T09:00:00Z",
                "snapshot_deadline_time": "2026-09-01T10:00:00Z",
                "player_id": 202,
                "club_id": 20,
                "price": 5.0,
            },
        ]
    )


def _stats() -> pd.DataFrame:
    rows = []
    player_101_points = [2, 0, 8, 4, 5, 6, 7, 8]
    for gameweek, points in enumerate(player_101_points, start=1):
        rows.append(
            {
                "season": "2026-27",
                "gameweek": gameweek,
                "player_id": 101,
                "total_points": points,
                "minutes": 90 if gameweek != 2 else 0,
            }
        )
        rows.append(
            {
                "season": "2026-27",
                "gameweek": gameweek,
                "player_id": 202,
                "total_points": 1,
                "minutes": 30,
            }
        )
    return pd.DataFrame(rows)


def test_past_features_and_future_labels_respect_boundary() -> None:
    result = build_training_table(_snapshots(), _stats()).set_index("player_id")

    # GW2 is an explicit blank (zero); GW3's 8 can represent an aggregate double.
    assert result.loc[101, "recent_points_mean_3"] == pytest.approx(10 / 3)
    assert result.loc[101, "recent_minutes_mean_3"] == pytest.approx(60)
    assert result.loc[101, "label_next_1_gameweek"] == 4
    assert result.loc[101, "label_minutes_next_1_gameweek"] == 90
    assert result.loc[101, "label_appearances_next_1_gameweek"] == 1
    assert result.loc[101, "label_starts_next_1_gameweek"] == 1
    assert result.loc[101, "label_next_3_gameweeks"] == 15
    assert result.loc[101, "label_minutes_next_3_gameweeks"] == 270
    assert result.loc[101, "label_appearances_next_3_gameweeks"] == 3
    assert result.loc[101, "label_starts_next_3_gameweeks"] == 3
    assert result.loc[101, "label_next_5_gameweeks"] == 30


def test_fixture_features_represent_blanks_and_doubles() -> None:
    fixtures = pd.DataFrame(
        [
            {
                "season": "2026-27",
                "snapshot_gameweek": 4,
                "gameweek_id": 4,
                "home_club_id": 10,
                "away_club_id": 20,
            },
            {
                "season": "2026-27",
                "snapshot_gameweek": 4,
                "gameweek_id": 4,
                "home_club_id": 30,
                "away_club_id": 10,
            },
            {
                "season": "2026-27",
                "snapshot_gameweek": 4,
                "gameweek_id": 6,
                "home_club_id": 20,
                "away_club_id": 30,
            },
        ]
    )
    result = build_training_table(
        _snapshots(), _stats(), fixture_snapshots=fixtures
    ).set_index("player_id")

    assert result.loc[101, "upcoming_fixture_count_1"] == 2
    assert result.loc[101, "upcoming_home_fixture_count_1"] == 1
    assert result.loc[101, "upcoming_away_fixture_count_1"] == 1
    assert result.loc[202, "upcoming_fixture_count_3"] == 2


def test_future_changes_do_not_change_past_features() -> None:
    baseline = build_training_table(_snapshots(), _stats())
    changed_stats = _stats()
    changed_stats.loc[
        (changed_stats["player_id"] == 101)
        & (changed_stats["gameweek"] == 4),
        "total_points",
    ] = 100
    changed = build_training_table(_snapshots(), changed_stats)

    assert baseline.loc[0, "recent_points_mean_3"] == changed.loc[
        0, "recent_points_mean_3"
    ]
    assert baseline.loc[0, "label_next_1_gameweek"] != changed.loc[
        0, "label_next_1_gameweek"
    ]


def test_incomplete_future_horizon_has_null_label() -> None:
    stats = _stats().loc[lambda frame: frame["gameweek"] <= 5]
    result = build_training_table(_snapshots(), stats).set_index("player_id")

    assert result.loc[101, "label_next_1_gameweek"] == 4
    assert pd.isna(result.loc[101, "label_next_3_gameweeks"])
    assert pd.isna(result.loc[101, "label_minutes_next_3_gameweeks"])
    assert pd.isna(result.loc[101, "label_appearances_next_3_gameweeks"])
    assert pd.isna(result.loc[101, "label_next_5_gameweeks"])


def test_missing_player_gameweek_does_not_become_zero() -> None:
    stats = _stats()
    stats = stats.loc[
        ~((stats["player_id"] == 101) & (stats["gameweek"] == 5))
    ]
    result = build_training_table(
        _snapshots(), stats, horizons=(3,)
    ).set_index("player_id")

    assert pd.isna(result.loc[101, "label_next_3_gameweeks"])
    assert result.loc[202, "label_next_3_gameweeks"] == 3


def test_post_deadline_training_row_is_rejected() -> None:
    snapshots = _snapshots()
    snapshots.loc[0, "snapshot_timestamp"] = "2026-09-01T10:00:01Z"

    with pytest.raises(FeatureValidationError, match="Post-deadline"):
        build_training_table(snapshots, _stats())


def test_empty_finalized_history_is_rejected() -> None:
    empty_stats = _stats().iloc[0:0]

    with pytest.raises(FeatureValidationError, match="no finalized rows"):
        build_training_table(_snapshots(), empty_stats)


def test_prediction_features_do_not_require_finalized_history() -> None:
    empty_stats = _stats().iloc[0:0]

    result = build_prediction_table(_snapshots(), empty_stats)

    assert len(result) == 2
    assert result["recent_points_mean_3"].isna().all()
    assert "label_next_1_gameweek" not in result
