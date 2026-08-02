"""DuckDB catalog over local immutable Parquet datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


def build_catalog(
    data_dir: Path,
    database_path: Path | None = None,
) -> Path:
    """Create or refresh a local DuckDB catalog and its deduplicated views."""
    database = database_path or data_dir / "fpl.duckdb"
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database))
    try:
        manifests = _read_snapshot_manifests(data_dir / "processed")
        connection.register("_snapshot_registry_input", manifests)
        connection.execute(
            "CREATE OR REPLACE TABLE snapshot_registry AS "
            "SELECT * FROM _snapshot_registry_input"
        )
        connection.unregister("_snapshot_registry_input")

        direct_root = data_dir / "processed"
        snapshot_tables = {
            "player_snapshots": "players.parquet",
            "club_snapshots": "clubs.parquet",
            "position_snapshots": "positions.parquet",
            "gameweek_snapshots": "gameweeks.parquet",
            "fixture_snapshots": "fixtures.parquet",
        }
        for view, filename in snapshot_tables.items():
            files = sorted(direct_root.glob(f"*/{filename}"))
            _create_parquet_view(connection, view, files)

        history_root = data_dir / "processed" / "history"
        stats_files = sorted(
            history_root.glob("*/*/player_gameweek_stats.parquet")
        )
        _create_parquet_view(connection, "_player_gameweek_stats_runs", stats_files)
        if stats_files:
            connection.execute(
                """
                CREATE OR REPLACE VIEW player_gameweek_stats AS
                SELECT * EXCLUDE (_row_number)
                FROM (
                    SELECT *,
                        row_number() OVER (
                            PARTITION BY season, gameweek, player_id
                            ORDER BY ingested_at DESC, filename DESC
                        ) AS _row_number
                    FROM _player_gameweek_stats_runs
                )
                WHERE _row_number = 1
                """
            )
        else:
            connection.execute(
                """
                CREATE OR REPLACE VIEW player_gameweek_stats AS
                SELECT
                    CAST(NULL AS VARCHAR) AS season,
                    CAST(NULL AS INTEGER) AS gameweek,
                    CAST(NULL AS BIGINT) AS player_id,
                    CAST(NULL AS DOUBLE) AS total_points,
                    CAST(NULL AS DOUBLE) AS minutes,
                    CAST(NULL AS VARCHAR) AS ingested_at,
                    CAST(NULL AS VARCHAR) AS filename
                WHERE false
                """
            )

        fixture_files = sorted(history_root.glob("*/*/fixtures.parquet"))
        _create_parquet_view(connection, "_historical_fixture_runs", fixture_files)
        if fixture_files:
            connection.execute(
                """
                CREATE OR REPLACE VIEW historical_fixtures AS
                SELECT * EXCLUDE (_row_number)
                FROM (
                    SELECT *,
                        row_number() OVER (
                            PARTITION BY season, fixture_id
                            ORDER BY ingested_at DESC, filename DESC
                        ) AS _row_number
                    FROM _historical_fixture_runs
                )
                WHERE _row_number = 1
                """
            )
        else:
            connection.execute(
                """
                CREATE OR REPLACE VIEW historical_fixtures AS
                SELECT
                    CAST(NULL AS VARCHAR) AS season,
                    CAST(NULL AS BIGINT) AS fixture_id,
                    CAST(NULL AS INTEGER) AS gameweek_id
                WHERE false
                """
            )

        manager_root = data_dir / "processed" / "manager_samples"
        rate_files = sorted(
            manager_root.glob("*/*/*/sample_player_rates.parquet")
        )
        _create_parquet_view(connection, "_manager_sample_rate_runs", rate_files)
        if rate_files:
            connection.execute(
                """
                CREATE OR REPLACE VIEW manager_sample_player_rates AS
                SELECT * EXCLUDE (_row_number)
                FROM (
                    SELECT *,
                        row_number() OVER (
                            PARTITION BY season, gameweek, player_id
                            ORDER BY collected_at DESC, filename DESC
                        ) AS _row_number
                    FROM _manager_sample_rate_runs
                )
                WHERE _row_number = 1
                """
            )
        else:
            connection.execute(
                """
                CREATE OR REPLACE VIEW manager_sample_player_rates AS
                SELECT
                    CAST(NULL AS VARCHAR) AS season,
                    CAST(NULL AS INTEGER) AS gameweek,
                    CAST(NULL AS BIGINT) AS player_id,
                    CAST(NULL AS DOUBLE) AS sample_ownership_percent
                WHERE false
                """
            )

        manager_files = sorted(manager_root.glob("*/*/*/managers.parquet"))
        _create_parquet_view(connection, "_manager_sample_runs", manager_files)
        if manager_files:
            connection.execute(
                """
                CREATE OR REPLACE VIEW manager_gameweek_samples AS
                SELECT * EXCLUDE (_row_number)
                FROM (
                    SELECT *,
                        row_number() OVER (
                            PARTITION BY season, gameweek, manager_id
                            ORDER BY collected_at DESC, filename DESC
                        ) AS _row_number
                    FROM _manager_sample_runs
                )
                WHERE _row_number = 1
                """
            )
        else:
            connection.execute(
                """
                CREATE OR REPLACE VIEW manager_gameweek_samples AS
                SELECT
                    CAST(NULL AS BIGINT) AS manager_id,
                    CAST(NULL AS VARCHAR) AS season,
                    CAST(NULL AS INTEGER) AS gameweek,
                    CAST(NULL AS VARCHAR) AS collected_at
                WHERE false
                """
            )

        pick_files = sorted(manager_root.glob("*/*/*/manager_picks.parquet"))
        _create_parquet_view(connection, "_manager_pick_runs", pick_files)
        if pick_files:
            connection.execute(
                """
                CREATE OR REPLACE VIEW manager_gameweek_picks AS
                SELECT * EXCLUDE (_row_number)
                FROM (
                    SELECT *,
                        row_number() OVER (
                            PARTITION BY season, gameweek, manager_id, player_id
                            ORDER BY collected_at DESC, filename DESC
                        ) AS _row_number
                    FROM _manager_pick_runs
                )
                WHERE _row_number = 1
                """
            )
        else:
            connection.execute(
                """
                CREATE OR REPLACE VIEW manager_gameweek_picks AS
                SELECT
                    CAST(NULL AS BIGINT) AS manager_id,
                    CAST(NULL AS VARCHAR) AS season,
                    CAST(NULL AS INTEGER) AS gameweek,
                    CAST(NULL AS BIGINT) AS player_id,
                    CAST(NULL AS INTEGER) AS squad_position
                WHERE false
                """
            )
    finally:
        connection.close()
    return database


def _create_parquet_view(
    connection: duckdb.DuckDBPyConnection,
    view_name: str,
    files: Iterable[Path],
) -> None:
    paths = list(files)
    if not paths:
        connection.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS "
            "SELECT CAST(NULL AS VARCHAR) AS filename WHERE false"
        )
        return
    path_literals = ", ".join(_sql_string(str(path.resolve())) for path in paths)
    connection.execute(
        f"CREATE OR REPLACE VIEW {view_name} AS "
        f"SELECT * FROM read_parquet([{path_literals}], "
        "union_by_name = true, filename = true)"
    )


def _read_snapshot_manifests(processed_dir: Path) -> pd.DataFrame:
    rows = []
    if processed_dir.exists():
        for path in sorted(processed_dir.glob("*/manifest.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata") or {}
            rows.append(
                {
                    "snapshot_id": payload.get("snapshot_id"),
                    "created_at_utc": payload.get("created_at_utc"),
                    "season": metadata.get("season"),
                    "snapshot_gameweek": metadata.get("snapshot_gameweek"),
                    "snapshot_timestamp": metadata.get("snapshot_timestamp"),
                    "snapshot_deadline_time": metadata.get(
                        "snapshot_deadline_time"
                    ),
                    "processed_dir": str(path.parent.resolve()),
                }
            )
    columns = [
        "snapshot_id",
        "created_at_utc",
        "season",
        "snapshot_gameweek",
        "snapshot_timestamp",
        "snapshot_deadline_time",
        "processed_dir",
    ]
    return pd.DataFrame(rows, columns=columns)


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
