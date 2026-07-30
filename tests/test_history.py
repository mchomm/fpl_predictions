"""Tests for finalized gameweek history ingestion."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

import pandas as pd
import pytest

from fpl_predictions.data.history import (
    ingest_finished_gameweeks,
    normalize_live_gameweek,
)
from fpl_predictions.data.normalize import NormalizationError


def _live_payload(points: int) -> dict:
    return {
        "elements": [
            {
                "id": 101,
                "stats": {"total_points": points, "minutes": 90},
                "explain": [],
            },
            {
                "id": 202,
                "stats": {"total_points": 0, "minutes": 0},
                "explain": [],
            },
        ]
    }


class HistoryClient:
    def __init__(self, bootstrap: dict, fixtures: list[dict]) -> None:
        self.bootstrap = bootstrap
        self.fixtures = fixtures

    def get_bootstrap(self) -> dict:
        return self.bootstrap

    def get_fixtures(self) -> list[dict]:
        return self.fixtures

    def get_live_gameweek(self, gameweek: int) -> dict:
        return _live_payload(gameweek * 2)


def test_live_gameweek_normalization_retains_aggregate_points() -> None:
    table = normalize_live_gameweek(
        _live_payload(8),
        "2026-27",
        3,
        datetime(2026, 9, 1, tzinfo=timezone.utc),
    )

    assert table.loc[0, "player_id"] == 101
    assert table.loc[0, "total_points"] == 8
    assert table.loc[0, "gameweek"] == 3


def test_history_ingestion_only_uses_finished_events(
    tmp_path,
    bootstrap_payload: dict,
    fixtures_payload: list[dict],
) -> None:
    bootstrap = deepcopy(bootstrap_payload)
    bootstrap["events"] = [
        {
            "id": 1,
            "name": "Gameweek 1",
            "deadline_time": "2026-08-15T10:00:00Z",
            "finished": True,
        },
        {
            "id": 2,
            "name": "Gameweek 2",
            "deadline_time": "2026-08-22T10:00:00Z",
            "finished": False,
        },
    ]
    result = ingest_finished_gameweeks(
        HistoryClient(bootstrap, fixtures_payload),  # type: ignore[arg-type]
        tmp_path,
        "2026-27",
        through_gameweek=1,
        ingested_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )

    assert result.gameweeks == (1,)
    stats = pd.read_parquet(
        result.processed_dir / "player_gameweek_stats.parquet"
    )
    assert stats["gameweek"].unique().tolist() == [1]
    raw = json.loads(
        (result.raw_dir / "live" / "gameweek-01.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw == _live_payload(2)


def test_history_rejects_unfinished_requested_gameweek(
    tmp_path,
    bootstrap_payload: dict,
    fixtures_payload: list[dict],
) -> None:
    bootstrap_payload["events"][0]["finished"] = False
    client = HistoryClient(bootstrap_payload, fixtures_payload)

    with pytest.raises(NormalizationError, match="No finished gameweeks"):
        ingest_finished_gameweeks(
            client,  # type: ignore[arg-type]
            tmp_path,
            "2026-27",
            through_gameweek=1,
        )

