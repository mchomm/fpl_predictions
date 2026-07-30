"""CLI for creating the local DuckDB catalog."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import duckdb

from fpl_predictions.config import Settings
from fpl_predictions.data.catalog import build_catalog


def main(argv: Sequence[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="fpl-build-catalog",
        description="Build DuckDB views over local FPL Parquet data.",
    )
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--database", type=Path)
    args = parser.parse_args(argv)
    try:
        path = build_catalog(args.data_dir, args.database)
    except (OSError, ValueError, duckdb.Error) as exc:
        parser.error(str(exc))
    print(f"Catalog: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
