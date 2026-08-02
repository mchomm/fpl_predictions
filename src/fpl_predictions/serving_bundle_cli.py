"""Build a compact, checksummed deployment bundle from selected artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

import pandas as pd

from fpl_predictions.data.storage import write_json, write_parquet
from fpl_predictions.squads.generation import generate_reference_population
from fpl_predictions.squads.rules import SquadRules


MODEL_ROLES = ("points", "minutes", "appearances", "starts")


def main(argv: Sequence[str] | None = None) -> int:
    """Create a self-contained current deployment bundle."""
    parser = argparse.ArgumentParser(prog="fpl-build-serving-bundle")
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-metadata", type=Path)
    parser.add_argument("--points-models", type=Path, required=True)
    parser.add_argument("--minutes-models", type=Path, required=True)
    parser.add_argument("--appearance-models", type=Path, required=True)
    parser.add_argument("--start-models", type=Path, required=True)
    parser.add_argument("--reference-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("deployment/current"))
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.reference_size <= 0:
        parser.error("reference-size must be positive")
    sources = [args.players, args.bootstrap, args.predictions]
    model_roots = {
        "points": args.points_models,
        "minutes": args.minutes_models,
        "appearances": args.appearance_models,
        "starts": args.start_models,
    }
    missing = [path for path in sources + list(model_roots.values()) if not path.exists()]
    if missing:
        parser.error("Missing inputs: " + ", ".join(map(str, missing)))

    players = pd.read_parquet(args.players)
    predictions = pd.read_parquet(args.predictions)
    bootstrap = json.loads(args.bootstrap.read_text(encoding="utf-8"))
    rules = SquadRules.from_bootstrap(bootstrap)
    horizons = sorted(
        int(column.removeprefix("predicted_points_"))
        for column in predictions
        if column.startswith("predicted_points_")
    )
    metadata = {}
    if args.prediction_metadata and args.prediction_metadata.is_file():
        metadata = json.loads(args.prediction_metadata.read_text(encoding="utf-8"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    )
    try:
        write_parquet(temporary / "players.parquet", players)
        write_parquet(temporary / "predictions.parquet", predictions)
        write_json(temporary / "bootstrap-static.json", bootstrap)
        reference_records = {}
        for horizon in horizons:
            references = generate_reference_population(
                players,
                predictions,
                rules,
                horizon,
                size=args.reference_size,
                strategy="human_like",
                seed=args.seed,
                target_cost=rules.budget,
                budget_band=1.0,
            )
            relative = Path("references") / f"horizon-{horizon}.parquet"
            (temporary / relative).parent.mkdir(parents=True, exist_ok=True)
            write_parquet(temporary / relative, references)
            reference_records[str(horizon)] = _record(temporary, relative)

        model_records = []
        for role, source_root in model_roots.items():
            artifact_directories = sorted(source_root.glob("horizon-*/artifact"))
            if not artifact_directories:
                raise ValueError(f"No artifacts found under {source_root}")
            for artifact in artifact_directories:
                horizon_name = artifact.parent.name
                for filename in ("model.joblib", "metadata.json", "feature_schema.json"):
                    source = artifact / filename
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    relative = Path("models") / role / horizon_name / filename
                    target = temporary / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    record = _record(temporary, relative)
                    record.update({"role": role, "horizon": horizon_name})
                    model_records.append(record)

        file_inventory = {
            "players": _record(temporary, Path("players.parquet")),
            "predictions": _record(temporary, Path("predictions.parquet")),
            "bootstrap": _record(temporary, Path("bootstrap-static.json")),
            "references": reference_records,
            "models": model_records,
        }
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "season": metadata.get("snapshot", {}).get("season"),
            "snapshot_gameweek": metadata.get("snapshot", {}).get(
                "snapshot_gameweek"
            ),
            "snapshot_timestamp": metadata.get("snapshot", {}).get(
                "snapshot_timestamp"
            ),
            "prediction_created_at_utc": metadata.get("created_at_utc"),
            "horizons": horizons,
            "reference_configuration": {
                "strategy": "human_like",
                "size": args.reference_size,
                "seed": args.seed,
                "budget_band": 1.0,
            },
            "model_runs": {
                role: source_root.name for role, source_root in model_roots.items()
            },
            "runtime_policy": {
                "predictions": "precomputed by the versioned model artifacts",
                "models_included": True,
                "training_data_included": False,
                "uploaded_images_persisted": False,
            },
            "files": file_inventory,
        }
        write_json(temporary / "manifest.json", manifest)
        temporary.rename(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    total_bytes = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    print(f"Serving bundle: {args.output}")
    print(f"Horizons: {horizons}")
    print(f"Size: {total_bytes / 1048576:.1f} MB")
    return 0


def _record(root: Path, relative: Path) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
