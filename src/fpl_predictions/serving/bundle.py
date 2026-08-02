"""Build-independent loading and verification of a deployed serving bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_predictions.squads.rules import SquadRules


class ServingBundleError(RuntimeError):
    """Raised when a deployment bundle is missing, stale, or corrupted."""


@dataclass(frozen=True, slots=True)
class ServingBundle:
    """All immutable runtime inputs required by the web application."""

    root: Path
    manifest: dict[str, Any]
    players: pd.DataFrame
    predictions: pd.DataFrame
    rules: SquadRules
    references: dict[int, pd.DataFrame]

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted(self.references))


def load_serving_bundle(
    root: Path,
    *,
    verify_checksums: bool = True,
) -> ServingBundle:
    """Load one self-describing serving directory and verify every artifact."""
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ServingBundleError(f"Serving manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ServingBundleError(f"Could not read {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise ServingBundleError(
            f"Unsupported serving schema: {manifest.get('schema_version')!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ServingBundleError("Serving manifest has no file inventory")
    if verify_checksums:
        _verify_files(root, files)

    players = pd.read_parquet(root / _path(files, "players"))
    predictions = pd.read_parquet(root / _path(files, "predictions"))
    bootstrap = json.loads(
        (root / _path(files, "bootstrap")).read_text(encoding="utf-8")
    )
    rules = SquadRules.from_bootstrap(bootstrap)
    reference_files = files.get("references")
    if not isinstance(reference_files, dict) or not reference_files:
        raise ServingBundleError("Serving bundle contains no reference populations")
    references = {
        int(horizon): pd.read_parquet(root / str(record["path"]))
        for horizon, record in reference_files.items()
    }
    _validate_tables(players, predictions, references)
    return ServingBundle(root, manifest, players, predictions, rules, references)


def _verify_files(root: Path, inventory: dict[str, Any]) -> None:
    records = []
    for key in ("players", "predictions", "bootstrap"):
        record = inventory.get(key)
        if not isinstance(record, dict):
            raise ServingBundleError(f"Missing serving file record: {key}")
        records.append(record)
    references = inventory.get("references")
    models = inventory.get("models")
    if not isinstance(references, dict) or not isinstance(models, list):
        raise ServingBundleError("Malformed references or models inventory")
    records.extend(references.values())
    records.extend(models)
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ServingBundleError("Malformed serving file inventory record")
        path = root / record["path"]
        if not path.is_file():
            raise ServingBundleError(f"Serving artifact is missing: {path}")
        expected_size = record.get("bytes")
        if expected_size != path.stat().st_size:
            raise ServingBundleError(f"Serving artifact size mismatch: {path}")
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != record.get("sha256"):
            raise ServingBundleError(f"Serving artifact checksum mismatch: {path}")


def _path(inventory: dict[str, Any], key: str) -> str:
    record = inventory.get(key)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ServingBundleError(f"Serving manifest has no valid {key} path")
    return str(record["path"])


def _validate_tables(
    players: pd.DataFrame,
    predictions: pd.DataFrame,
    references: dict[int, pd.DataFrame],
) -> None:
    player_required = {
        "player_id",
        "display_name",
        "club_id",
        "club_name",
        "position_id",
        "position_short_name",
        "price",
    }
    missing = player_required.difference(players.columns)
    if missing:
        raise ServingBundleError(
            "Serving players are missing: " + ", ".join(sorted(missing))
        )
    if players["player_id"].duplicated().any():
        raise ServingBundleError("Serving players contain duplicate IDs")
    if predictions["player_id"].duplicated().any():
        raise ServingBundleError("Serving predictions contain duplicate IDs")
    prediction_ids = set(predictions["player_id"])
    if not prediction_ids.issubset(set(players["player_id"])):
        raise ServingBundleError("Serving predictions contain unknown player IDs")
    for horizon, frame in references.items():
        point_column = f"predicted_points_{horizon}"
        if point_column not in predictions:
            raise ServingBundleError(
                f"Predictions do not cover reference horizon {horizon}"
            )
        if frame.empty or "overall_points" not in frame:
            raise ServingBundleError(
                f"Reference population for horizon {horizon} is invalid"
            )
