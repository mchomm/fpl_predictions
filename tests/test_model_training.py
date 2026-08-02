"""Integration tests for temporal training and saved artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from fpl_predictions.modelling.artifacts import load_artifact
from fpl_predictions.modelling.predict import predict_players
from fpl_predictions.modelling.train import train_horizon


def _training_table() -> pd.DataFrame:
    rows = []
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    for gameweek in range(1, 11):
        for player_id in range(1, 13):
            recent = float((player_id + gameweek) % 8)
            points = 0.8 * recent + 0.2 * (player_id % 4)
            rows.append(
                {
                    "season": "2026-27",
                    "snapshot_gameweek": gameweek,
                    "snapshot_timestamp": start
                    + timedelta(days=7 * gameweek),
                    "snapshot_deadline_time": start
                    + timedelta(days=7 * gameweek, hours=1),
                    "player_id": player_id,
                    "display_name": f"Player {player_id}",
                    "price": 4.0 + player_id / 10,
                    "form": recent,
                    "total_points": gameweek * recent,
                    "minutes": gameweek * 80,
                    "recent_points_mean_3": recent,
                    "recent_minutes_mean_3": 80.0,
                    "upcoming_fixture_count_1": 1,
                    "position_short_name": ["GKP", "DEF", "MID", "FWD"][
                        player_id % 4
                    ],
                    "status": "a",
                    "club_id": player_id % 3,
                    "label_next_1_gameweek": points,
                    "label_minutes_next_1_gameweek": 90.0,
                    "recent_appearance_rate_3": float(player_id % 2),
                    "recent_start_rate_3": float(player_id % 3 == 0),
                    "label_appearances_next_1_gameweek": float(player_id % 2),
                    "label_starts_next_1_gameweek": float(player_id % 3 == 0),
                }
            )
    return pd.DataFrame(rows)


def test_train_compare_save_load_and_predict(tmp_path) -> None:
    table = _training_table()
    result = train_horizon(
        table,
        horizon=1,
        output_dir=tmp_path / "horizon-1",
        min_train_periods=3,
        calibration_bins=4,
        random_seed=7,
        source_paths=[],
    )

    assert result.model_path.exists()
    assert set(result.leaderboard["model"]) == {
        "historical_mean",
        "recent_points_mean",
        "ridge",
        "random_forest",
    }
    assert result.predictions_path.exists()
    assert result.calibration_path.exists()

    artifact = load_artifact(result.model_path)
    predictions = predict_players(artifact, table.iloc[:3])

    assert artifact.metadata["horizon_gameweeks"] == 1
    assert artifact.metadata["training_rows"] == len(table)
    assert artifact.metadata["evaluation"]["folds"] <= 12
    assert predictions["predicted_future_points"].notna().all()
    assert predictions["prediction_horizon"].unique().tolist() == [1]
    assert "player_id" not in artifact.schema.columns


def test_train_expected_minutes_artifact(tmp_path) -> None:
    table = _training_table()
    result = train_horizon(
        table,
        horizon=1,
        output_dir=tmp_path / "minutes-horizon-1",
        min_train_periods=3,
        calibration_bins=4,
        random_seed=7,
        source_paths=[],
        target_kind="minutes",
    )

    artifact = load_artifact(result.model_path)
    predictions = predict_players(artifact, table.iloc[:3])

    assert artifact.metadata["target_kind"] == "minutes"
    assert artifact.metadata["target"] == "label_minutes_next_1_gameweek"
    assert {
        "recent_minutes_mean",
        "recent_minutes_with_model_fallback",
    }.issubset(set(result.leaderboard["model"]))
    assert predictions["predicted_future_points"].notna().all()


def test_train_calibrated_appearance_probability(tmp_path) -> None:
    table = _training_table()
    result = train_horizon(
        table,
        horizon=1,
        output_dir=tmp_path / "appearance-horizon-1",
        min_train_periods=3,
        calibration_bins=4,
        random_seed=7,
        source_paths=[],
        target_kind="appearances",
    )

    artifact = load_artifact(result.model_path)
    predictions = predict_players(artifact, table.iloc[:6])

    assert artifact.metadata["target_kind"] == "appearances"
    assert artifact.metadata["probability_calibration"]["audit_rows"] > 0
    assert predictions["predicted_future_points"].between(0, 1).all()
