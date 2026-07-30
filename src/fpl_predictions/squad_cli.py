"""CLI for validating, projecting, and rating a complete FPL squad."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import duckdb
import pandas as pd

from fpl_predictions.config import Settings
from fpl_predictions.data.catalog import build_catalog
from fpl_predictions.data.storage import utc_snapshot_id, write_json, write_parquet
from fpl_predictions.squads.generation import (
    REFERENCE_STRATEGIES,
    generate_reference_population,
)
from fpl_predictions.squads.projection import project_squad
from fpl_predictions.squads.ratings import rate_squad
from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.schemas import SquadSelection
from fpl_predictions.squads.validation import validate_squad


def main(argv: Sequence[str] | None = None) -> int:
    """Rate a submitted squad against reproducible legal reference squads."""
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="fpl-rate-squad",
        description="Validate and rate a complete FPL squad from JSON.",
    )
    parser.add_argument("--squad", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--reference-size", type=int, default=1000)
    parser.add_argument(
        "--strategy",
        choices=sorted(REFERENCE_STRATEGIES),
        default="price_aware",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        prediction_path = args.predictions or _latest_predictions(
            args.data_dir.parent / "outputs" / "predictions"
        )
        predictions = pd.read_parquet(prediction_path)
        players, bootstrap, snapshot = _load_snapshot_context(
            args.data_dir,
            args.database,
            predictions,
        )
        rules = SquadRules.from_bootstrap(bootstrap)
        squad_payload = json.loads(args.squad.read_text(encoding="utf-8"))
        if not isinstance(squad_payload, dict):
            raise ValueError("Squad JSON must contain one object")
        selection = SquadSelection.from_mapping(squad_payload)
        validated = validate_squad(selection, players, rules)
        projection = project_squad(
            validated, predictions, rules, args.horizon
        )
        references = generate_reference_population(
            players=players,
            predictions=predictions,
            rules=rules,
            horizon=args.horizon,
            size=args.reference_size,
            strategy=args.strategy,
            seed=args.seed,
        )
        reference_name = (
            f"{args.strategy} legal squads (horizon={args.horizon}, "
            f"seed={args.seed})"
        )
        rating = rate_squad(projection, references, reference_name)

        if args.output:
            report_path = args.output
            references_path = args.output.with_name(
                f"{args.output.stem}-references.parquet"
            )
        else:
            run_id = utc_snapshot_id(datetime.now(timezone.utc))
            output_dir = args.data_dir.parent / "outputs" / "ratings" / run_id
            report_path = output_dir / "rating.json"
            references_path = output_dir / "references.parquet"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot": snapshot,
            "prediction_file": str(prediction_path.resolve()),
            "squad": selection.as_dict(),
            "validation": {
                "formation": validated.formation,
                "total_cost": validated.total_cost,
                "budget_limit": validated.budget_limit,
            },
            "rules": asdict(rules),
            "projection": projection.as_dict(),
            "rating": rating.as_dict(),
            "reference_configuration": {
                "strategy": args.strategy,
                "size": args.reference_size,
                "seed": args.seed,
                "horizon": args.horizon,
                "percentile_method": "empirical midrank",
            },
            "reference_file": str(references_path.resolve()),
        }
        write_parquet(references_path, references)
        write_json(report_path, report)
    except (
        duckdb.Error,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))

    print(f"Formation: {validated.formation}")
    print(f"Overall rating: {rating.scores['overall']:.1f}/100")
    print(f"Report: {report_path}")
    print(f"References: {references_path}")
    return 0


def _latest_predictions(directory: Path) -> Path:
    paths = sorted(directory.glob("players-*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No prediction Parquet files found under {directory}"
        )
    return paths[-1]


def _load_snapshot_context(
    data_dir: Path,
    database_path: Path | None,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    required = {
        "season",
        "snapshot_gameweek",
        "snapshot_timestamp",
        "player_id",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            "Prediction table is missing identity columns: "
            + ", ".join(sorted(missing))
        )
    keys = predictions.loc[
        :, ["season", "snapshot_gameweek", "snapshot_timestamp"]
    ].drop_duplicates()
    if len(keys) != 1:
        raise ValueError("Prediction table must describe exactly one snapshot")
    season, gameweek, raw_timestamp = keys.iloc[0].tolist()
    timestamp = pd.Timestamp(raw_timestamp).isoformat()
    database = build_catalog(data_dir, database_path)
    with duckdb.connect(str(database), read_only=True) as connection:
        registry = connection.execute(
            "SELECT snapshot_id, processed_dir FROM snapshot_registry "
            "WHERE season = ? AND snapshot_gameweek = ? "
            "AND snapshot_timestamp = ?",
            [season, int(gameweek), timestamp],
        ).fetchone()
        if registry is None:
            raise ValueError(
                "No saved snapshot matches the prediction table identity"
            )
        players = connection.execute(
            "SELECT * EXCLUDE (filename) FROM player_snapshots "
            "WHERE season = ? AND snapshot_gameweek = ? "
            "AND snapshot_timestamp = ?",
            [season, int(gameweek), timestamp],
        ).fetchdf()
    bootstrap_path = data_dir / "raw" / str(registry[0]) / "bootstrap-static.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    return players, bootstrap, {
        "snapshot_id": str(registry[0]),
        "season": str(season),
        "snapshot_gameweek": int(gameweek),
        "snapshot_timestamp": str(timestamp),
    }


if __name__ == "__main__":
    raise SystemExit(main())
