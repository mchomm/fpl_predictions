"""CLI for applying horizon artifacts to the latest current player snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import duckdb
import pandas as pd

from fpl_predictions.config import Settings
from fpl_predictions.data.catalog import build_catalog
from fpl_predictions.data.features import (
    FeatureValidationError,
    build_prediction_table,
)
from fpl_predictions.data.storage import (
    SnapshotStorageError,
    utc_snapshot_id,
    write_json,
    write_parquet,
)
from fpl_predictions.modelling.artifacts import load_artifact
from fpl_predictions.modelling.features import ModelFeatureError
from fpl_predictions.modelling.predict import predict_players
from fpl_predictions.modelling.lineup_roles import reconcile_lineup_probabilities


def main(argv: Sequence[str] | None = None) -> int:
    """Build current features and generate one prediction column per artifact."""
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="fpl-predict-players",
        description="Predict current player points from saved horizon artifacts.",
    )
    parser.add_argument("--models-run", type=Path, required=True)
    parser.add_argument(
        "--minutes-models-run",
        type=Path,
        help="Optional expected-minutes model run for matching horizons.",
    )
    parser.add_argument(
        "--appearance-models-run",
        type=Path,
        help="Optional calibrated next-gameweek appearance model run.",
    )
    parser.add_argument(
        "--start-models-run",
        type=Path,
        help="Optional calibrated next-gameweek start model run.",
    )
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--season")
    parser.add_argument("--gameweek", type=int)
    parser.add_argument("--recent-window", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        artifact_paths = sorted(args.models_run.glob("horizon-*/artifact"))
        if not artifact_paths:
            raise FileNotFoundError(
                f"No horizon artifacts found under {args.models_run}"
            )
        artifacts = [load_artifact(path) for path in artifact_paths]
        if any(
            artifact.metadata.get("target_kind", "points") != "points"
            for artifact in artifacts
        ):
            raise ValueError("--models-run must contain points artifacts")
        minutes_artifacts = []
        if args.minutes_models_run is not None:
            minute_paths = sorted(
                args.minutes_models_run.glob("horizon-*/artifact")
            )
            if not minute_paths:
                raise FileNotFoundError(
                    "No horizon artifacts found under "
                    f"{args.minutes_models_run}"
                )
            minutes_artifacts = [load_artifact(path) for path in minute_paths]
            if any(
                artifact.metadata.get("target_kind") != "minutes"
                for artifact in minutes_artifacts
            ):
                raise ValueError(
                    "--minutes-models-run must contain minutes artifacts"
                )
        probability_artifacts: dict[str, object] = {}
        for target_kind, run_path in (
            ("appearances", args.appearance_models_run),
            ("starts", args.start_models_run),
        ):
            if run_path is None:
                continue
            paths = sorted(run_path.glob("horizon-*/artifact"))
            if len(paths) != 1:
                raise ValueError(
                    f"--{target_kind}-models-run must contain exactly one artifact"
                )
            artifact = load_artifact(paths[0])
            if (
                artifact.metadata.get("target_kind") != target_kind
                or int(artifact.metadata.get("horizon_gameweeks", 0)) != 1
            ):
                raise ValueError(
                    f"--{target_kind}-models-run must contain a horizon-1 "
                    f"{target_kind} artifact"
                )
            probability_artifacts[target_kind] = artifact
        horizons = sorted(
            {
            int(artifact.metadata["horizon_gameweeks"])
            for artifact in artifacts + minutes_artifacts
            }
            | {1 for _ in probability_artifacts}
        )
        database = build_catalog(args.data_dir, args.database)
        players, fixtures, stats, snapshot = _load_snapshot_data(
            database,
            args.season,
            args.gameweek,
        )
        features = build_prediction_table(
            players,
            stats,
            fixture_snapshots=fixtures,
            horizons=horizons,
            recent_window=args.recent_window,
        )
        identity_columns = [
            column
            for column in (
                "season",
                "snapshot_gameweek",
                "snapshot_timestamp",
                "player_id",
                "club_id",
                "display_name",
                "position_id",
                "position_short_name",
                "club_name",
                "price",
                "status",
                "chance_of_playing_this_round",
                "chance_of_playing_next_round",
                "news",
                "recent_gameweeks_available",
            )
            if column in features
        ]
        predictions = features.loc[:, identity_columns].copy()
        for artifact in artifacts:
            horizon = int(artifact.metadata["horizon_gameweeks"])
            predicted = predict_players(artifact, features)
            predictions[f"predicted_points_{horizon}"] = predicted[
                "predicted_future_points"
            ].to_numpy()
        for artifact in minutes_artifacts:
            horizon = int(artifact.metadata["horizon_gameweeks"])
            predicted = predict_players(artifact, features)
            predictions[f"predicted_minutes_{horizon}"] = predicted[
                "predicted_future_points"
            ].clip(lower=0, upper=90 * horizon).to_numpy()
        for target_kind, artifact in probability_artifacts.items():
            predicted = predict_players(artifact, features)
            output_name = (
                "appearance_probability_1"
                if target_kind == "appearances"
                else "start_probability_1"
            )
            predictions[output_name] = predicted[
                "predicted_future_points"
            ].clip(lower=0, upper=1).to_numpy()
        if {
            "appearance_probability_1",
            "start_probability_1",
        }.issubset(predictions.columns):
            predictions["start_probability_1"] = predictions[
                ["start_probability_1", "appearance_probability_1"]
            ].min(axis=1)
            predictions, lineup_role_audit = reconcile_lineup_probabilities(
                predictions
            )
        else:
            lineup_role_audit = None

        output = args.output or (
            args.data_dir.parent
            / "outputs"
            / "predictions"
            / f"players-{utc_snapshot_id(datetime.now(timezone.utc))}.parquet"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(output, predictions)
        write_json(
            output.with_suffix(".json"),
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "snapshot": snapshot,
                "models_run": str(args.models_run.resolve()),
                "minutes_models_run": (
                    str(args.minutes_models_run.resolve())
                    if args.minutes_models_run is not None
                    else None
                ),
                "appearance_models_run": (
                    str(args.appearance_models_run.resolve())
                    if args.appearance_models_run is not None
                    else None
                ),
                "start_models_run": (
                    str(args.start_models_run.resolve())
                    if args.start_models_run is not None
                    else None
                ),
                "horizons": horizons,
                "rows": len(predictions),
                "lineup_role_reconciliation": lineup_role_audit,
                "limitations": [
                    "Historical models do not yet use archived injury status.",
                    "Preseason recent-form features may be unavailable.",
                    "Predictions are raw player expectations, not squad ratings.",
                ],
            },
        )
    except (
        FeatureValidationError,
        FileNotFoundError,
        ModelFeatureError,
        SnapshotStorageError,
        OSError,
        ValueError,
        duckdb.Error,
    ) as exc:
        parser.error(str(exc))

    print(
        f"Snapshot: {snapshot['season']} GW{snapshot['snapshot_gameweek']}"
    )
    print(f"Players: {len(predictions)}")
    print(f"Predictions: {output}")
    return 0


def _load_snapshot_data(
    database: Path,
    season: str | None,
    gameweek: int | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    with duckdb.connect(str(database), read_only=True) as connection:
        clauses = ["season IS NOT NULL", "snapshot_gameweek IS NOT NULL"]
        parameters: list[object] = []
        if season is not None:
            clauses.append("season = ?")
            parameters.append(season)
        if gameweek is not None:
            clauses.append("snapshot_gameweek = ?")
            parameters.append(gameweek)
        where = " AND ".join(clauses)
        row = connection.execute(
            "SELECT season, snapshot_gameweek, snapshot_timestamp "
            "FROM snapshot_registry "
            f"WHERE {where} "
            "ORDER BY snapshot_timestamp DESC LIMIT 1",
            parameters,
        ).fetchone()
        if row is None:
            raise ValueError("No matching deadline snapshot was found")
        snapshot = {
            "season": str(row[0]),
            "snapshot_gameweek": int(row[1]),
            "snapshot_timestamp": str(row[2]),
        }
        snapshot_parameters = [row[0], row[1], row[2]]
        snapshot_filter = (
            "season = ? AND snapshot_gameweek = ? "
            "AND snapshot_timestamp = ?"
        )
        players = connection.execute(
            "SELECT * EXCLUDE (filename) FROM player_snapshots WHERE "
            + snapshot_filter,
            snapshot_parameters,
        ).fetchdf()
        fixtures = connection.execute(
            "SELECT * EXCLUDE (filename) FROM fixture_snapshots WHERE "
            + snapshot_filter,
            snapshot_parameters,
        ).fetchdf()
        stats = connection.execute(
            "SELECT * EXCLUDE (filename) FROM player_gameweek_stats "
            "WHERE season = ? AND gameweek < ?",
            [row[0], row[1]],
        ).fetchdf()
    return players, fixtures, stats, snapshot


if __name__ == "__main__":
    raise SystemExit(main())
