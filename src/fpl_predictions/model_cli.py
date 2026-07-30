"""CLI for temporal model comparison and artifact training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Sequence

from fpl_predictions.config import Settings
from fpl_predictions.data.training import (
    TrainingTableError,
    load_training_tables,
)
from fpl_predictions.data.storage import utc_snapshot_id, write_json
from fpl_predictions.modelling.features import ModelFeatureError
from fpl_predictions.modelling.train import train_horizon
from fpl_predictions.modelling.validation import TemporalValidationError


def build_parser() -> argparse.ArgumentParser:
    """Create the model training parser."""
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="fpl-train-models",
        description="Compare and train future FPL player-points models.",
    )
    parser.add_argument(
        "--training-data",
        type=Path,
        nargs="+",
        help="Training Parquet file; defaults to the newest local table.",
    )
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--min-train-gameweeks", type=int, default=4)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--max-validation-folds", type=int, default=12)
    parser.add_argument("--minimum-feature-coverage", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Train and save one selected model per requested horizon."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        training_paths = args.training_data or [
            _latest_training_table(args.data_dir)
        ]
        table = load_training_tables(training_paths)
        run_id = utc_snapshot_id(datetime.now(timezone.utc))
        run_dir = args.models_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        summaries = []
        for horizon in args.horizons:
            result = train_horizon(
                table,
                horizon,
                run_dir / f"horizon-{horizon}",
                min_train_periods=args.min_train_gameweeks,
                calibration_bins=args.calibration_bins,
                random_seed=args.random_seed,
                source_paths=training_paths,
                max_validation_folds=args.max_validation_folds,
                minimum_feature_coverage=args.minimum_feature_coverage,
            )
            best = result.leaderboard.iloc[0]
            summaries.append(
                {
                    "horizon": horizon,
                    "selected_model": result.selected_model,
                    "mae": float(best["mae"]),
                    "rmse": float(best["rmse"]),
                    "model_path": str(result.model_path.resolve()),
                }
            )
        write_json(
            run_dir / "run.json",
            {
                "run_id": run_id,
                "training_data": [
                    str(path.resolve()) for path in training_paths
                ],
                "horizons": summaries,
            },
        )
    except (
        FileNotFoundError,
        ModelFeatureError,
        TemporalValidationError,
        TrainingTableError,
        OSError,
        ValueError,
    ) as exc:
        parser.error(str(exc))

    print(f"Run: {run_id}")
    for summary in summaries:
        print(
            f"Horizon {summary['horizon']}: "
            f"{summary['selected_model']} "
            f"(MAE {summary['mae']:.3f}, RMSE {summary['rmse']:.3f})"
        )
    print(f"Artifacts: {run_dir}")
    return 0


def _latest_training_table(data_dir: Path) -> Path:
    files = list((data_dir / "processed" / "training").glob("training-*.parquet"))
    if not files:
        raise FileNotFoundError(
            "No training table found; run fpl-build-training-data first"
        )
    return max(files, key=lambda path: path.stat().st_mtime)


if __name__ == "__main__":
    raise SystemExit(main())
