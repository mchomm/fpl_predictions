"""CLI for ingesting finalized official gameweek history."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from fpl_predictions.api.client import FPLAPIError, FPLClient
from fpl_predictions.config import Settings
from fpl_predictions.data.history import ingest_finished_gameweeks
from fpl_predictions.data.normalize import NormalizationError
from fpl_predictions.data.storage import SnapshotStorageError


def build_parser() -> argparse.ArgumentParser:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="fpl-ingest-history",
        description="Ingest finalized FPL live gameweek statistics.",
    )
    parser.add_argument("--season", required=True)
    parser.add_argument("--through-gameweek", type=int)
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
        result = ingest_finished_gameweeks(
            FPLClient(args.base_url, args.timeout),
            args.data_dir,
            args.season,
            args.through_gameweek,
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
    print(f"Gameweeks: {result.gameweeks[0]}-{result.gameweeks[-1]}")
    print(f"Parquet: {result.processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

