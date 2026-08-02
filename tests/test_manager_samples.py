"""Tests for privacy-minimized post-deadline manager sampling."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import duckdb
import pandas as pd
import pytest

from fpl_predictions.data.catalog import build_catalog
from fpl_predictions.data.managers import (
    collect_manager_sample,
    discover_overall_league_id,
    normalize_manager_picks,
)
from fpl_predictions.data.normalize import NormalizationError


def _bootstrap() -> dict:
    position_ids = [1] * 2 + [2] * 5 + [3] * 5 + [4] * 3
    return {
        "elements": [
            {
                "id": value,
                "team": (value - 1) % 5 + 1,
                "element_type": position_ids[value - 1],
                "now_cost": 50,
                "selected_by_percent": str(value),
            }
            for value in range(1, 16)
        ],
        "teams": [],
        "element_types": [
            {"id": 1, "singular_name_short": "GKP"},
            {"id": 2, "singular_name_short": "DEF"},
            {"id": 3, "singular_name_short": "MID"},
            {"id": 4, "singular_name_short": "FWD"},
        ],
        "total_players": 100,
        "events": [
            {
                "id": 1,
                "deadline_time": "2026-08-21T17:30:00Z",
            }
        ],
    }


def _picks_payload(points: int = 10) -> dict:
    ordered_ids = [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 2, 6, 7, 15]
    return {
        "active_chip": None,
        "entry_history": {
            "event": 1,
            "points": points,
            "total_points": points,
            "overall_rank": 25,
            "bank": 0,
            "value": 1000,
        },
        "picks": [
            {
                "element": player_id,
                "position": position,
                "multiplier": 2 if player_id == 13 else (
                    1 if position <= 11 else 0
                ),
                "is_captain": player_id == 13,
                "is_vice_captain": player_id == 14,
            }
            for position, player_id in enumerate(ordered_ids, start=1)
        ],
        "automatic_subs": [],
    }


class FakeManagerClient:
    def __init__(self) -> None:
        self.pick_calls: list[tuple[int, int]] = []

    def get_bootstrap(self) -> dict:
        return _bootstrap()

    def get_manager_picks(self, manager_id: int, gameweek: int) -> dict:
        self.pick_calls.append((manager_id, gameweek))
        return _picks_payload(points=manager_id)

    def get_manager(self, manager_id: int) -> dict:
        return {
            "id": manager_id,
            "leagues": {
                "classic": [
                    {"id": 7, "name": "Overall", "short_name": "overall"}
                ]
            },
        }

    def get_classic_league_standings(self, league_id: int, page: int) -> dict:
        start = (page - 1) * 50 + 1
        return {
            "standings": {
                "results": [
                    {
                        "entry": value,
                        "rank": value,
                        "total": 0,
                        "event_total": 0,
                    }
                    for value in range(start, min(start + 50, 101))
                ]
            }
        }


def test_discover_overall_league_id() -> None:
    client = FakeManagerClient()
    assert discover_overall_league_id(client, 1) == 7  # type: ignore[arg-type]


def test_normalize_manager_picks_validates_and_labels_lineup() -> None:
    picks, manager = normalize_manager_picks(
        _picks_payload(),
        manager_id=42,
        season="2026-27",
        gameweek=1,
        collected_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
        valid_player_ids=set(range(1, 16)),
    )
    assert len(picks) == 15
    assert picks["lineup_role"].value_counts().to_dict() == {
        "starter": 11,
        "bench": 4,
    }
    assert picks.loc[picks["is_captain"], "player_id"].item() == 13
    assert manager["overall_rank"] == 25


def test_collection_refuses_predeadline_access(tmp_path) -> None:
    client = FakeManagerClient()
    with pytest.raises(NormalizationError, match="cannot be collected before"):
        collect_manager_sample(
            client,  # type: ignore[arg-type]
            tmp_path,
            "2026-27",
            1,
            manager_ids=[10],
            collected_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
            request_interval_seconds=0,
        )
    assert client.pick_calls == []


def test_explicit_collection_storage_and_catalog(tmp_path) -> None:
    client = FakeManagerClient()
    result = collect_manager_sample(
        client,  # type: ignore[arg-type]
        tmp_path,
        "2026-27",
        1,
        manager_ids=[10, 20],
        collected_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
        request_interval_seconds=0,
    )
    managers = pd.read_parquet(result.processed_dir / "managers.parquet")
    picks = pd.read_parquet(result.processed_dir / "manager_picks.parquet")
    manifest = json.loads(
        (result.processed_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert result.managers == 2
    assert len(picks) == 30
    assert set(managers["manager_id"]) == {10, 20}
    assert manifest["sample_method"] == "explicit"
    assert manifest["tables"]["sample_player_rates"] == 15
    assert "names are intentionally excluded" in manifest["privacy"]
    audit = json.loads(
        (result.processed_dir / "audit.json").read_text(encoding="utf-8")
    )
    assert audit["formations"] == {"3-5-2": 2}
    assert audit["current_squad_cost"]["median"] == 75.0

    database = build_catalog(tmp_path)
    with duckdb.connect(str(database), read_only=True) as connection:
        manager_count = connection.execute(
            "SELECT count(*) FROM manager_gameweek_samples"
        ).fetchone()[0]
        pick_count = connection.execute(
            "SELECT count(*) FROM manager_gameweek_picks"
        ).fetchone()[0]
        rate_count = connection.execute(
            "SELECT count(*) FROM manager_sample_player_rates"
        ).fetchone()[0]
    assert manager_count == 2
    assert pick_count == 30
    assert rate_count == 15


def test_overall_sample_is_seeded_and_privacy_minimized(tmp_path) -> None:
    result = collect_manager_sample(
        FakeManagerClient(),  # type: ignore[arg-type]
        tmp_path,
        "2026-27",
        1,
        sample_size=10,
        random_seed=9,
        collected_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
        request_interval_seconds=0,
    )
    frame = json.loads(
        (result.raw_dir / "sampling-frame.json").read_text(encoding="utf-8")
    )
    assert len(frame) == 10
    assert all("manager_id" in row and "player_name" not in row for row in frame)
    manifest = json.loads(
        (result.processed_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["overall_league_id"] == 7
    assert manifest["sample_method"] == "uniform_overall_pages"
