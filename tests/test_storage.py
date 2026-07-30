"""Tests for reproducible local snapshot storage."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd

from fpl_predictions.data.normalize import normalize_current_data
from fpl_predictions.data.storage import save_current_snapshot


def test_snapshot_writes_exact_json_and_parquet(
    tmp_path,
    bootstrap_payload: dict,
    fixtures_payload: list[dict],
) -> None:
    tables = normalize_current_data(bootstrap_payload, fixtures_payload)
    timestamp = datetime(2026, 7, 30, 14, 5, 6, tzinfo=timezone.utc)

    location = save_current_snapshot(
        tmp_path,
        bootstrap_payload,
        fixtures_payload,
        tables,
        timestamp,
    )

    assert location.snapshot_id == "20260730T140506000000Z"
    with (location.raw_dir / "bootstrap-static.json").open(
        encoding="utf-8"
    ) as handle:
        assert json.load(handle) == bootstrap_payload

    stored_players = pd.read_parquet(location.processed_dir / "players.parquet")
    assert stored_players["player_id"].tolist() == [101, 202]
    with (location.processed_dir / "manifest.json").open(
        encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    assert manifest["tables"]["players"] == 2


def test_snapshot_json_encodes_nested_values_for_parquet(
    tmp_path,
    bootstrap_payload: dict,
    fixtures_payload: list[dict],
) -> None:
    bootstrap_payload["element_types"][0]["rules"] = {}
    bootstrap_payload["element_types"][1]["rules"] = {"clean_sheets": 4}
    tables = normalize_current_data(bootstrap_payload, fixtures_payload)

    location = save_current_snapshot(
        tmp_path,
        bootstrap_payload,
        fixtures_payload,
        tables,
        datetime(2026, 7, 30, 15, 0, tzinfo=timezone.utc),
    )

    positions = pd.read_parquet(location.processed_dir / "positions.parquet")
    assert positions.loc[0, "rules"] == "{}"
    assert positions.loc[1, "rules"] == '{"clean_sheets": 4}'
