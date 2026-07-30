"""Command-line entry points."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from fpl_predictions.api.client import FPLAPIError, FPLClient
from fpl_predictions.config import Settings
from fpl_predictions.data.normalize import NormalizationError
from fpl_predictions.data.refresh import refresh_current_data
from fpl_predictions.data.storage import SnapshotStorageError


def build_parser() -> argparse.ArgumentParser:
    """Create the current-data refresh argument parser."""
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="fpl-refresh",
        description="Fetch and store a current FPL API snapshot.",
    )
    parser.add_argument("--base-url", default=settings.base_url)
    parser.add_argument("--timeout", type=float, default=settings.timeout_seconds)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument(
        "--verbose", action="store_true", help="Enable detailed logging."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a current-data refresh, returning a process exit status."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        client = FPLClient(args.base_url, args.timeout)
        location = refresh_current_data(client, args.data_dir)
    except (
        FPLAPIError,
        NormalizationError,
        SnapshotStorageError,
        OSError,
        ValueError,
    ) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    print(f"Snapshot: {location.snapshot_id}")
    print(f"Raw JSON: {location.raw_dir}")
    print(f"Parquet: {location.processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
