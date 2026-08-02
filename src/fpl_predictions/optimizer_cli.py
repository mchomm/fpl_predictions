"""CLI for exact squad construction and transfer recommendations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import duckdb
import pandas as pd

from fpl_predictions.config import Settings
from fpl_predictions.data.storage import utc_snapshot_id, write_json
from fpl_predictions.squad_cli import _latest_predictions, _load_snapshot_context
from fpl_predictions.squads.optimizer import (
    SquadOptimizationError,
    optimize_squad,
)
from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.schemas import SquadSelection
from fpl_predictions.squads.simulation import simulate_selection


def main(argv: Sequence[str] | None = None) -> int:
    """Optimize a new squad or bounded transfers from an existing squad."""
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="fpl-optimize-squad",
        description="Build an exact projected-points-optimal legal FPL squad.",
    )
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--squad", type=Path, help="Existing squad JSON to improve")
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument(
        "--max-transfers",
        type=int,
        default=1,
        help="Maximum changes when --squad is supplied (default: 1).",
    )
    parser.add_argument("--budget", type=float)
    parser.add_argument("--bench-weight", type=float, default=0.1)
    parser.add_argument(
        "--free-transfers",
        type=int,
        default=1,
        help="Saved free transfers available, from 0 to 5 (default: 1).",
    )
    parser.add_argument(
        "--hit-cost",
        type=float,
        default=4.0,
        help="Points deducted per transfer beyond the free allowance (default: 4).",
    )
    parser.add_argument("--lock", type=int, nargs="*", default=[])
    parser.add_argument("--exclude", type=int, nargs="*", default=[])
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument(
        "--availability-simulations",
        type=int,
        default=0,
        help="Optional post-optimization scenario evaluation; not the objective.",
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
        current = None
        if args.squad is not None:
            payload = json.loads(args.squad.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Squad JSON must contain one object")
            current = SquadSelection.from_mapping(payload)
        result = optimize_squad(
            players,
            predictions,
            rules,
            horizon=args.horizon,
            current_squad=current,
            max_transfers=(args.max_transfers if current is not None else None),
            budget=args.budget,
            bench_weight=args.bench_weight,
            locked_player_ids=args.lock,
            excluded_player_ids=args.exclude,
            time_limit_seconds=args.time_limit,
            free_transfers=args.free_transfers,
            hit_cost=args.hit_cost,
        )
        simulation = None
        if args.availability_simulations:
            prediction_columns = [
                column
                for column in predictions.columns
                if column not in players.columns or column == "player_id"
            ]
            simulation_players = players.merge(
                predictions.loc[:, prediction_columns],
                on="player_id",
                how="inner",
                validate="one_to_one",
            )
            simulation = simulate_selection(
                simulation_players,
                result.selection,
                rules,
                args.horizon,
                simulations=args.availability_simulations,
                seed=2026,
            )

        if args.output is not None:
            output = args.output
        else:
            run_id = utc_snapshot_id(datetime.now(timezone.utc))
            output = (
                args.data_dir.parent
                / "outputs"
                / "optimizations"
                / run_id
                / "optimization.json"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot": snapshot,
            "prediction_file": str(prediction_path.resolve()),
            "configuration": {
                "horizon": args.horizon,
                "max_transfers": (
                    args.max_transfers if current is not None else None
                ),
                "budget_override": args.budget,
                "bench_weight": args.bench_weight,
                "free_transfers": (
                    args.free_transfers if current is not None else None
                ),
                "hit_cost": args.hit_cost if current is not None else None,
                "locked_player_ids": args.lock,
                "excluded_player_ids": args.exclude,
                "time_limit_seconds": args.time_limit,
                "objective": (
                    "starting-XI points + captain bonus + bench_weight × bench points"
                ),
            },
            "current_squad": current.as_dict() if current is not None else None,
            "optimization": result.as_dict(players, predictions),
            "availability_simulation": (
                simulation.as_dict() if simulation is not None else None
            ),
            "limitations": [
                "Exact affordability requires selling_prices (or purchase_prices) "
                "in the squad JSON; screenshots do not expose manager-specific "
                "sell values or bank balance.",
                "The selected horizon values waiting as well as acting now, but "
                "this command does not optimize a separate transfer sequence for "
                "each future Gameweek.",
                "Wildcard, Free Hit, Bench Boost, and Triple Captain planning is "
                "outside this optimizer.",
                "Availability simulation is optional post-optimization evidence, "
                "not yet a nonlinear optimization objective.",
            ],
        }
        write_json(output, report)
    except (
        duckdb.Error,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        SquadOptimizationError,
        ValueError,
    ) as exc:
        parser.error(str(exc))

    print(f"Formation: {result.projection.formation}")
    print(f"Projected points: {result.projection.overall_points:.3f}")
    if result.projected_points_gain is not None:
        print(f"Gross projected gain: {result.projected_points_gain:+.3f}")
        print(
            "Net projected gain after hits: "
            f"{result.net_projected_points_gain:+.3f}"
        )
    if result.transfers_in:
        indexed = players.set_index("player_id")
        incoming = ", ".join(
            str(indexed.loc[value, "display_name"]) for value in result.transfers_in
        )
        outgoing = ", ".join(
            str(indexed.loc[value, "display_name"]) for value in result.transfers_out
        )
        print(f"Transfers: {outgoing} -> {incoming}")
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
