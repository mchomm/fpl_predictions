"""CLI for collecting privacy-minimized public manager squad samples."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from fpl_predictions.api.client import FPLAPIError, FPLClient
from fpl_predictions.config import Settings
from fpl_predictions.data.managers import collect_manager_sample
from fpl_predictions.data.normalize import NormalizationError
from fpl_predictions.data.storage import SnapshotStorageError


def build_parser() -> argparse.ArgumentParser:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="fpl-snapshot-managers",
        description=(
            "Collect a reproducible public manager-squad sample after a deadline."
        ),
    )
    parser.add_argument("--season", required=True)
    parser.add_argument("--gameweek", required=True, type=int)
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--discovery-manager-id", type=int, default=1)
    parser.add_argument(
        "--manager-id",
        action="append",
        type=int,
        dest="manager_ids",
        help=(
            "Collect an explicit public manager ID; repeat to bypass random "
            "Overall sampling."
        ),
    )
    parser.add_argument("--request-interval", type=float, default=0.1)
    parser.add_argument("--base-url", default=settings.base_url)
    parser.add_argument("--timeout", type=float, default=settings.timeout_seconds)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        result = collect_manager_sample(
            FPLClient(args.base_url, args.timeout),
            args.data_dir,
            args.season,
            args.gameweek,
            sample_size=args.sample_size,
            random_seed=args.seed,
            discovery_manager_id=args.discovery_manager_id,
            manager_ids=args.manager_ids,
            request_interval_seconds=args.request_interval,
        )
    except (
        FPLAPIError,
        NormalizationError,
        SnapshotStorageError,
        OSError,
        ValueError,
    ) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    print(f"Ingestion: {result.ingestion_id}")
    print(f"Managers: {result.managers}")
    print(f"Parquet: {result.processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
