"""CLI for historical feature and label reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fpl_predictions.config import Settings
from fpl_predictions.data.backfill import (
    BackfillValidationError,
    build_historical_backfill,
)
from fpl_predictions.data.storage import SnapshotStorageError
from fpl_predictions.sources.vaastav import HistoricalSourceError


def main(argv: Sequence[str] | None = None) -> int:
    """Build an audited historical training table from a pinned download."""
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="fpl-build-historical-training",
        description="Reconstruct leakage-conscious historical training rows.",
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seasons", nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--recent-window", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        result = build_historical_backfill(
            args.source_root,
            args.data_dir,
            args.seasons,
            args.horizons,
            args.recent_window,
        )
    except (
        BackfillValidationError,
        HistoricalSourceError,
        SnapshotStorageError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(f"Backfill: {result.ingestion_id}")
    print(f"Seasons: {', '.join(result.seasons)}")
    print(f"Training table: {result.training_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

