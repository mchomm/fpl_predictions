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
from fpl_predictions.squads.projection import (
    player_projection_details,
    project_squad,
)
from fpl_predictions.squads.ratings import rate_squad
from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.schemas import SquadSelection
from fpl_predictions.squads.simulation import simulate_selection
from fpl_predictions.squads.validation import ValidatedSquad, validate_squad


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
        default="human_like",
    )
    parser.add_argument(
        "--reference-budget-band",
        type=float,
        default=1.0,
        help=(
            "For human_like references, spend within this many millions of "
            "the official budget (default: 1.0)."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--availability-simulations",
        type=int,
        default=None,
        help=(
            "Appearance scenarios per squad for autosubs and vice-captain "
            "takeover. By default, 500 are enabled once the submitted squad "
            "has at least two recent gameweeks; use 0 for deterministic mode."
        ),
    )
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
        simulation_count, simulation_readiness = _simulation_count(
            args.availability_simulations,
            validated,
            predictions,
        )
        deterministic_projection = project_squad(
            validated, predictions, rules, args.horizon
        )
        simulation = None
        projection = deterministic_projection
        if simulation_count:
            prediction_columns = [
                column
                for column in predictions.columns
                if column not in validated.player_rows.columns
                or column == "player_id"
            ]
            simulation_players = validated.player_rows.merge(
                predictions.loc[:, prediction_columns],
                on="player_id",
                how="left",
                validate="one_to_one",
            )
            simulation = simulate_selection(
                simulation_players,
                selection,
                rules,
                args.horizon,
                simulations=simulation_count,
                seed=args.seed,
            )
            projection = simulation.projection
        references = generate_reference_population(
            players=players,
            predictions=predictions,
            rules=rules,
            horizon=args.horizon,
            size=args.reference_size,
            strategy=args.strategy,
            seed=args.seed,
            target_cost=rules.budget,
            budget_band=args.reference_budget_band,
            availability_simulations=simulation_count,
        )
        if args.strategy == "human_like":
            reference_name = (
                "ownership-conditioned, budget-matched legal reference "
                f"population (horizon={args.horizon}, seed={args.seed})"
            )
        else:
            reference_name = (
                f"{args.strategy} legal reference population "
                f"(horizon={args.horizon}, seed={args.seed})"
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
            "deterministic_projection_audit": deterministic_projection.as_dict(),
            "availability_simulation": (
                simulation.as_dict() if simulation is not None else None
            ),
            "player_projections": player_projection_details(
                validated,
                predictions,
                args.horizon,
            ),
            "rating": rating.as_dict(),
            "reference_configuration": {
                "strategy": args.strategy,
                "size": args.reference_size,
                "seed": args.seed,
                "horizon": args.horizon,
                "percentile_method": "empirical midrank",
                "selection_prior": (
                    "product of current player ownership percentages, with a "
                    "0.1 percentage-point measurement-resolution floor; "
                    "conditional-independence approximation"
                    if args.strategy == "human_like"
                    else args.strategy
                ),
                "target_cost": (
                    rules.budget if args.strategy == "human_like" else None
                ),
                "budget_band": (
                    args.reference_budget_band
                    if args.strategy == "human_like"
                    else None
                ),
                "lineup_policy": (
                    "model-best legal XI and model-best captain from each squad"
                ),
                "availability_simulations_per_squad": (
                    simulation_count
                ),
                "availability_simulation_seed": args.seed,
                "availability_simulation_readiness": simulation_readiness,
            },
            "reference_summary": _reference_summary(references, players),
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


def _simulation_count(
    requested: int | None,
    validated: ValidatedSquad,
    predictions: pd.DataFrame,
) -> tuple[int, dict[str, object]]:
    """Gate automatic simulation until participation features have live history."""
    if requested is not None and requested < 0:
        raise ValueError("availability simulations must be non-negative")
    selected_ids = list(validated.selection.player_ids)
    selected = predictions[predictions["player_id"].isin(selected_ids)]
    coverage = pd.to_numeric(
        selected.get(
            "recent_gameweeks_available",
            pd.Series(float("nan"), index=selected.index),
        ),
        errors="coerce",
    )
    median_history = (
        float(coverage.median()) if coverage.notna().any() else 0.0
    )
    ready = median_history >= 2.0
    resolved = requested if requested is not None else (500 if ready else 0)
    return resolved, {
        "automatic_gate": requested is None,
        "ready": ready,
        "median_recent_gameweeks_available": median_history,
        "minimum_for_automatic_simulation": 2,
        "note": (
            "Simulation enabled."
            if resolved
            else "Deterministic projection retained until live participation "
            "history is sufficient; pass --availability-simulations to test "
            "the cold-start scenarios explicitly."
        ),
    }


def _reference_summary(
    references: pd.DataFrame,
    players: pd.DataFrame,
) -> dict[str, object]:
    """Expose the comparison distribution so high ratings remain auditable."""
    costs = references["total_cost"]
    overall = references["overall_points"]
    summary: dict[str, object] = {
        "cost": {
            "minimum": float(costs.min()),
            "median": float(costs.median()),
            "maximum": float(costs.max()),
        },
        "overall_points": {
            "p10": float(overall.quantile(0.10)),
            "median": float(overall.median()),
            "p90": float(overall.quantile(0.90)),
            "p95": float(overall.quantile(0.95)),
            "p99": float(overall.quantile(0.99)),
            "maximum": float(overall.max()),
        },
    }
    if "ownership_percent" in players and "player_ids" in references:
        selected_ids = references["player_ids"].map(json.loads).explode()
        generated_percent = (
            selected_ids.value_counts() * 100.0 / len(references)
        )
        calibration = players.loc[
            :, ["player_id", "ownership_percent"]
        ].copy()
        calibration["ownership_percent"] = pd.to_numeric(
            calibration["ownership_percent"], errors="coerce"
        )
        calibration["generated_percent"] = (
            calibration["player_id"].map(generated_percent).fillna(0.0)
        )
        valid = calibration.dropna(subset=["ownership_percent"])
        correlation = (
            valid[["ownership_percent", "generated_percent"]]
            .corr()
            .iloc[0, 1]
        )
        summary["ownership_calibration"] = {
            "pearson_correlation": (
                float(correlation) if pd.notna(correlation) else None
            ),
            "mean_absolute_error_percentage_points": float(
                (
                    valid["ownership_percent"]
                    - valid["generated_percent"]
                )
                .abs()
                .mean()
            ),
            "note": (
                "Generated marginal inclusion versus current API ownership; "
                "budget and legality conditioning can shift individual rates."
            ),
        }
    return summary


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
