"""Tests for leakage-conscious historical reconstruction."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_predictions.data.backfill import (
    _future_window_sum,
    _past_window_mean,
    build_historical_backfill,
    reconstruct_season,
)

SEASON_DIR = Path(__file__).parent / "fixtures" / "vaastav" / "2023-24"


def test_reconstruction_shifts_performance_and_excludes_xp() -> None:
    table = reconstruct_season(
        SEASON_DIR,
        "2023-24",
        horizons=(1, 3),
        recent_window=3,
    ).set_index(["player_id", "snapshot_gameweek"])

    assert "xP" not in table.columns
    assert table.loc[(101, 1), "total_points"] == 0
    assert table.loc[(101, 2), "total_points"] == 2
    assert table.loc[(101, 2), "recent_points_mean_3"] == 2
    assert table.loc[(101, 2), "price"] == pytest.approx(4.5)
    assert table.loc[(101, 4), "gameweek_total_points"] == 8
    assert table.loc[(101, 4), "upcoming_fixture_count_1"] == 2
    assert table.loc[(101, 4), "label_next_1_gameweek"] == 8
    assert table.loc[(101, 4), "label_minutes_next_1_gameweek"] == 180
    assert table.loc[(101, 3), "label_next_3_gameweeks"] == 16
    assert table.loc[(101, 1), "club_attack_strength"] == pytest.approx(1.0)
    assert table.loc[(101, 4), "club_strength_matches"] == 3
    assert "upcoming_opponent_attack_strength_mean_1" in table.columns


def test_future_fixture_results_do_not_change_past_strength(tmp_path) -> None:
    baseline = reconstruct_season(SEASON_DIR, "2023-24", horizons=(1,))
    copied = tmp_path / "2023-24"
    copied.mkdir()
    (copied / "gws").mkdir()
    for relative in ("teams.csv", "gws/merged_gw.csv"):
        source = SEASON_DIR / relative
        target = copied / relative
        target.write_bytes(source.read_bytes())
    fixtures = pd.read_csv(SEASON_DIR / "fixtures.csv")
    fixtures.loc[fixtures["event"] == 4, ["team_h_score", "team_a_score"]] = 99
    fixtures.to_csv(copied / "fixtures.csv", index=False)

    changed = reconstruct_season(copied, "2023-24", horizons=(1,))
    columns = [column for column in baseline if "strength" in column]
    earlier = baseline["snapshot_gameweek"] <= 4
    pd.testing.assert_frame_equal(
        baseline.loc[earlier, columns].reset_index(drop=True),
        changed.loc[earlier, columns].reset_index(drop=True),
    )


def test_blank_and_double_gameweeks_are_explicit() -> None:
    table = reconstruct_season(
        SEASON_DIR,
        "2023-24",
        horizons=(1,),
    ).set_index(["player_id", "snapshot_gameweek"])

    assert table.loc[(101, 4), "fixture_count"] == 2
    assert table.loc[(202, 4), "fixture_count"] == 0
    assert table.loc[(202, 4), "label_next_1_gameweek"] == 0
    assert pd.notna(table.loc[(202, 4), "snapshot_timestamp"])


def test_backfill_writes_audit_manifest_and_training_table(tmp_path) -> None:
    source_root = SEASON_DIR.parent
    result = build_historical_backfill(
        source_root,
        tmp_path,
        ["2023-24"],
        horizons=(1, 3),
        verify_source=False,
        created_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )

    training = pd.read_parquet(result.training_table)
    assert len(training) == 12
    assert result.player_gameweeks.is_file()
    assert (result.directory / "audit.json").is_file()
    manifest = json.loads(
        (result.directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_verified"] is False
    assert "xP is excluded" in " ".join(manifest["limitations"])


def test_league_wide_blank_is_zero_but_missing_player_data_is_unknown() -> None:
    frame = pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2],
            "snapshot_gameweek": [1, 3, 1, 3],
            "points": [2.0, 4.0, 1.0, pd.NA],
        }
    )

    future = _future_window_sum(frame, "points", 3, {2})
    recent, available = _past_window_mean(frame, "points", 2, {2})

    assert future.iloc[0] == 6
    assert pd.isna(future.iloc[2])
    assert recent.iloc[1] == 1
    assert available.iloc[1] == 2
