"""CLI for constructing a leakage-safe local training table."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import duckdb

from fpl_predictions.config import Settings
from fpl_predictions.data.catalog import build_catalog
from fpl_predictions.data.features import (
    FeatureValidationError,
    build_training_table,
)
from fpl_predictions.data.storage import (
    SnapshotStorageError,
    utc_snapshot_id,
    write_parquet,
)


def main(argv: Sequence[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="fpl-build-training-data",
        description="Build past-only features and future FPL points labels.",
    )
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--recent-window", type=int, default=3)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        database = build_catalog(args.data_dir, args.database)
        with duckdb.connect(str(database), read_only=True) as connection:
            snapshots = connection.execute(
                "SELECT * EXCLUDE (filename) FROM player_snapshots "
                "WHERE season IS NOT NULL AND snapshot_gameweek IS NOT NULL"
            ).fetchdf()
            stats = connection.execute(
                "SELECT * EXCLUDE (filename) FROM player_gameweek_stats"
            ).fetchdf()
            fixtures = connection.execute(
                "SELECT * EXCLUDE (filename) FROM fixture_snapshots "
                "WHERE season IS NOT NULL AND snapshot_gameweek IS NOT NULL"
            ).fetchdf()
        training = build_training_table(
            snapshots,
            stats,
            fixture_snapshots=fixtures,
            horizons=args.horizons,
            recent_window=args.recent_window,
        )
        output = args.output or (
            args.data_dir
            / "processed"
            / "training"
            / f"training-{utc_snapshot_id(datetime.now(timezone.utc))}.parquet"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(output, training)
    except (
        FeatureValidationError,
        SnapshotStorageError,
        OSError,
        ValueError,
        duckdb.Error,
    ) as exc:
        parser.error(str(exc))

    print(f"Rows: {len(training)}")
    print(f"Training table: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
