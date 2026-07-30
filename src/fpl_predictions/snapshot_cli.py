"""CLI for collecting a pre-deadline gameweek snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Sequence

from fpl_predictions.api.client import FPLAPIError, FPLClient
from fpl_predictions.config import Settings
from fpl_predictions.data.normalize import NormalizationError
from fpl_predictions.data.refresh import refresh_current_data
from fpl_predictions.data.snapshots import (
    SnapshotValidationError,
    snapshot_from_bootstrap,
)
from fpl_predictions.data.storage import SnapshotStorageError


def build_parser() -> argparse.ArgumentParser:
    """Create the gameweek snapshot parser."""
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="fpl-snapshot-gameweek",
        description="Save a deadline-validated FPL gameweek snapshot.",
    )
    parser.add_argument("--season", required=True, help="Season label, e.g. 2026-27")
    parser.add_argument("--gameweek", required=True, type=int)
    parser.add_argument("--base-url", default=settings.base_url)
    parser.add_argument("--timeout", type=float, default=settings.timeout_seconds)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Collect and save a deadline-safe gameweek snapshot."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        client = FPLClient(args.base_url, args.timeout)
        bootstrap = client.get_bootstrap()
        fixtures = client.get_fixtures()
        snapshot = snapshot_from_bootstrap(
            bootstrap,
            args.season,
            args.gameweek,
            collected_at=datetime.now(timezone.utc),
        )
        location = refresh_current_data(
            client,
            args.data_dir,
            snapshot,
            bootstrap=bootstrap,
            fixtures=fixtures,
        )
    except (
        FPLAPIError,
        NormalizationError,
        SnapshotStorageError,
        SnapshotValidationError,
        OSError,
        ValueError,
    ) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    print(f"Snapshot: {location.snapshot_id}")
    print(f"Gameweek: {args.season} GW{args.gameweek}")
    print(f"Parquet: {location.processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
