"""Tests for DuckDB catalog discovery and deduplication."""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pandas as pd

from fpl_predictions.data.catalog import build_catalog
from fpl_predictions.data.normalize import normalize_current_data
from fpl_predictions.data.snapshots import (
    attach_snapshot_metadata,
    snapshot_from_bootstrap,
)
from fpl_predictions.data.storage import (
    save_current_snapshot,
    write_parquet,
)


def test_catalog_exposes_snapshots_and_latest_history_run(
    tmp_path,
    bootstrap_payload: dict,
    fixtures_payload: list[dict],
) -> None:
    snapshot = snapshot_from_bootstrap(
        bootstrap_payload,
        "2026-27",
        1,
        datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    tables = attach_snapshot_metadata(
        normalize_current_data(bootstrap_payload, fixtures_payload), snapshot
    )
    save_current_snapshot(
        tmp_path,
        bootstrap_payload,
        fixtures_payload,
        tables,
        timestamp=snapshot.snapshot_timestamp,
        metadata=snapshot.as_metadata(),
    )

    history_root = tmp_path / "processed" / "history" / "2026-27"
    for run, ingested_at, points in (
        ("run-1", "2026-08-20T10:00:00Z", 2),
        ("run-2", "2026-08-21T10:00:00Z", 5),
    ):
        run_dir = history_root / run
        run_dir.mkdir(parents=True)
        write_parquet(
            run_dir / "player_gameweek_stats.parquet",
            pd.DataFrame(
                [
                    {
                        "season": "2026-27",
                        "gameweek": 1,
                        "player_id": 101,
                        "total_points": points,
                        "minutes": 90,
                        "ingested_at": ingested_at,
                    }
                ]
            ),
        )

    database = build_catalog(tmp_path)
    with duckdb.connect(str(database), read_only=True) as connection:
        snapshot_count = connection.execute(
            "SELECT count(*) FROM player_snapshots"
        ).fetchone()[0]
        latest_points = connection.execute(
            "SELECT total_points FROM player_gameweek_stats"
        ).fetchone()[0]
        registry = connection.execute(
            "SELECT season, snapshot_gameweek FROM snapshot_registry"
        ).fetchone()

    assert snapshot_count == 2
    assert latest_points == 5
    assert registry == ("2026-27", 1)

