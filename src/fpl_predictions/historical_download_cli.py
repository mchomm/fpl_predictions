"""CLI for reproducible download of archived historical FPL data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fpl_predictions.config import Settings
from fpl_predictions.sources.vaastav import (
    HistoricalSourceError,
    download_seasons,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Download required historical files with checksums and provenance."""
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="fpl-download-history",
        description="Download pinned Vaastav historical FPL seasons.",
    )
    parser.add_argument("--seasons", nargs="+", required=True)
    parser.add_argument(
        "--revision",
        required=True,
        help="Complete 40-character source Git commit SHA.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=settings.data_dir / "raw" / "external",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    try:
        source_root = download_seasons(
            args.destination,
            args.seasons,
            args.revision,
            args.timeout,
        )
    except (
        FileExistsError,
        HistoricalSourceError,
        OSError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(f"Historical source: {source_root}")
    print(f"Seasons: {', '.join(args.seasons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

