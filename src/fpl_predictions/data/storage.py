"""Local raw JSON and normalized Parquet snapshot storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class SnapshotLocation:
    """Paths created for one current-data refresh."""

    snapshot_id: str
    raw_dir: Path
    processed_dir: Path


class SnapshotStorageError(RuntimeError):
    """Raised when a local snapshot cannot be written completely."""


def utc_snapshot_id(timestamp: datetime | None = None) -> str:
    """Return a filesystem-safe UTC identifier with microsecond precision."""
    value = timestamp or datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("Snapshot timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def save_current_snapshot(
    data_dir: Path,
    bootstrap: dict[str, Any],
    fixtures: list[dict[str, Any]],
    tables: dict[str, pd.DataFrame],
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> SnapshotLocation:
    """Save exact API payloads and normalized tables under one snapshot ID."""
    snapshot_id = utc_snapshot_id(timestamp)
    raw_dir = data_dir / "raw" / snapshot_id
    processed_dir = data_dir / "processed" / snapshot_id
    raw_dir.mkdir(parents=True, exist_ok=False)
    processed_dir.mkdir(parents=True, exist_ok=False)

    try:
        _atomic_json_write(raw_dir / "bootstrap-static.json", bootstrap)
        _atomic_json_write(raw_dir / "fixtures.json", fixtures)
        for name, table in tables.items():
            _atomic_parquet_write(processed_dir / f"{name}.parquet", table)

        manifest = {
            "snapshot_id": snapshot_id,
            "created_at_utc": datetime.strptime(
                snapshot_id, "%Y%m%dT%H%M%S%fZ"
            )
            .replace(tzinfo=timezone.utc)
            .isoformat(),
            "raw_files": ["bootstrap-static.json", "fixtures.json"],
            "tables": {name: len(table) for name, table in tables.items()},
            "metadata": metadata or {},
        }
        _atomic_json_write(processed_dir / "manifest.json", manifest)
    except Exception as exc:
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(processed_dir, ignore_errors=True)
        if isinstance(exc, SnapshotStorageError):
            raise
        raise SnapshotStorageError(
            f"Could not save FPL snapshot {snapshot_id}: {exc}"
        ) from exc
    return SnapshotLocation(snapshot_id, raw_dir, processed_dir)


def _atomic_json_write(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    """Atomically write a JSON value to an existing parent directory."""
    _atomic_json_write(path, value)


def _atomic_parquet_write(path: Path, table: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        _parquet_safe_table(table).to_parquet(temporary, index=False)
        temporary.replace(path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise SnapshotStorageError(
            f"Could not write Parquet table {path.name}: {exc}"
        ) from exc


def write_parquet(path: Path, table: pd.DataFrame) -> None:
    """Atomically write a DataFrame, encoding nested JSON values safely."""
    _atomic_parquet_write(path, table)


def _parquet_safe_table(table: pd.DataFrame) -> pd.DataFrame:
    """Encode nested JSON values that have no stable tabular Parquet type."""
    result = table.copy()
    for column in result.select_dtypes(include=["object"]).columns:
        values = result[column]
        has_nested_value = values.map(
            lambda value: isinstance(value, (dict, list))
        ).any()
        if has_nested_value:
            result[column] = values.map(
                lambda value: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
            )
    return result
