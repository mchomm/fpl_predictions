"""Temporal model comparison and final artifact training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from fpl_predictions.data.storage import write_json, write_parquet
from fpl_predictions.modelling.artifacts import (
    ModelArtifact,
    ProbabilityCalibratedRegressor,
    save_artifact,
)
from fpl_predictions.modelling.evaluation import (
    calibration_by_prediction_bins,
    prediction_metrics,
)
from fpl_predictions.modelling.features import (
    FeatureSchema,
    infer_feature_schema,
    prepare_features,
)
from fpl_predictions.modelling.models import candidate_models
from fpl_predictions.modelling.validation import (
    TemporalFold,
    rolling_origin_folds,
)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Outputs and selected model for one future-points horizon."""

    horizon: int
    selected_model: str
    model_path: Path
    leaderboard: pd.DataFrame
    predictions_path: Path
    calibration_path: Path


def label_column(horizon: int, target_kind: str = "points") -> str:
    """Return the training label name for a future-gameweek horizon."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if target_kind not in {"points", "minutes", "appearances", "starts"}:
        raise ValueError(
            "target_kind must be points, minutes, appearances, or starts"
        )
    suffix = "gameweek" if horizon == 1 else "gameweeks"
    prefix = "" if target_kind == "points" else f"{target_kind}_"
    return f"label_{prefix}next_{horizon}_{suffix}"


def train_horizon(
    training_table: pd.DataFrame,
    horizon: int,
    output_dir: Path,
    min_train_periods: int = 4,
    calibration_bins: int = 10,
    random_seed: int = 42,
    source_paths: Sequence[Path] | None = None,
    max_validation_folds: int | None = 12,
    minimum_feature_coverage: float = 0.5,
    target_kind: str = "points",
) -> TrainingResult:
    """Compare candidates temporally and save the best fitted model."""
    if target_kind in {"appearances", "starts"} and horizon != 1:
        raise ValueError(
            f"{target_kind} probability artifacts currently require horizon 1"
        )
    label = label_column(horizon, target_kind)
    if label not in training_table:
        raise ValueError(f"Training table has no label column {label!r}")
    data = training_table.dropna(subset=[label]).copy()
    if data.empty:
        raise ValueError(f"No finalized labels are available for horizon {horizon}")
    data[label] = pd.to_numeric(data[label], errors="raise")
    data["snapshot_timestamp"] = pd.to_datetime(
        data["snapshot_timestamp"], utc=True, errors="raise"
    )

    schema = infer_feature_schema(data, minimum_feature_coverage)
    features = prepare_features(data, schema)
    target = data[label].astype(float)
    eligible_folds = rolling_origin_folds(data, horizon, min_train_periods)
    folds = _select_folds(eligible_folds, max_validation_folds)
    models = candidate_models(schema, horizon, random_seed, target_kind)

    prediction_frames: list[pd.DataFrame] = []
    for model_name, estimator in models.items():
        for fold in folds:
            fitted = clone(estimator)
            fitted.fit(
                features.loc[fold.train_index],
                target.loc[fold.train_index],
            )
            predicted = fitted.predict(features.loc[fold.validation_index])
            identity_columns = [
                column
                for column in (
                    "season",
                    "snapshot_gameweek",
                    "snapshot_timestamp",
                    "player_id",
                )
                if column in data
            ]
            fold_predictions = data.loc[
                fold.validation_index, identity_columns
            ].copy()
            fold_predictions["fold"] = fold.fold
            fold_predictions["model"] = model_name
            fold_predictions["actual"] = target.loc[
                fold.validation_index
            ].to_numpy()
            fold_predictions["prediction"] = predicted
            prediction_frames.append(fold_predictions)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metric_rows = []
    calibration_frames = []
    for model_name, model_predictions in predictions.groupby("model"):
        metrics = prediction_metrics(model_predictions)
        metric_rows.append({"model": model_name, **metrics})
        calibration = calibration_by_prediction_bins(
            model_predictions, calibration_bins
        )
        calibration.insert(0, "model", model_name)
        calibration_frames.append(calibration)
    leaderboard = pd.DataFrame(metric_rows).sort_values(
        ["mae", "rmse", "model"], ignore_index=True
    )
    selected_name = str(leaderboard.iloc[0]["model"])
    selected_model = clone(models[selected_name])
    selected_model.fit(features, target)
    probability_calibration = None
    if target_kind in {"appearances", "starts"}:
        selected_rows = predictions.loc[
            predictions["model"] == selected_name
        ].copy()
        calibrator, probability_calibration = _fit_probability_calibrator(
            selected_rows
        )
        selected_rows["calibrated_prediction"] = calibrator.predict_proba(
            selected_rows[["prediction"]].to_numpy(dtype=float)
        )[:, 1]
        predictions = predictions.merge(
            selected_rows[
                [
                    "fold",
                    "model",
                    "player_id",
                    "snapshot_timestamp",
                    "calibrated_prediction",
                ]
            ],
            on=[
                "fold",
                "model",
                "player_id",
                "snapshot_timestamp",
            ],
            how="left",
            validate="one_to_one",
        )
        selected_model = ProbabilityCalibratedRegressor(
            selected_model,
            calibrator,
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "fold_predictions.parquet"
    calibration_path = output_dir / "calibration.parquet"
    write_parquet(predictions_path, predictions)
    calibration_table = pd.concat(calibration_frames, ignore_index=True)
    write_parquet(calibration_path, calibration_table)
    write_parquet(output_dir / "leaderboard.parquet", leaderboard)

    metadata = _metadata(
        data=data,
        horizon=horizon,
        label=label,
        schema=schema,
        selected_model=selected_name,
        leaderboard=leaderboard,
        folds=len(folds),
        eligible_folds=len(eligible_folds),
        min_train_periods=min_train_periods,
        random_seed=random_seed,
        source_paths=source_paths,
        minimum_feature_coverage=minimum_feature_coverage,
        target_kind=target_kind,
        probability_calibration=probability_calibration,
    )
    write_json(output_dir / "evaluation.json", metadata["evaluation"])
    artifact = ModelArtifact(selected_model, schema, metadata)
    model_path = save_artifact(artifact, output_dir / "artifact")
    return TrainingResult(
        horizon,
        selected_name,
        model_path,
        leaderboard,
        predictions_path,
        calibration_path,
    )


def _metadata(
    data: pd.DataFrame,
    horizon: int,
    label: str,
    schema: FeatureSchema,
    selected_model: str,
    leaderboard: pd.DataFrame,
    folds: int,
    eligible_folds: int,
    min_train_periods: int,
    random_seed: int,
    source_paths: Sequence[Path] | None,
    minimum_feature_coverage: float,
    target_kind: str,
    probability_calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    evaluation = {
        "folds": folds,
        "eligible_folds": eligible_folds,
        "min_train_periods": min_train_periods,
        "leaderboard": [
            {
                key: _json_value(value)
                for key, value in row.items()
            }
            for row in leaderboard.to_dict(orient="records")
        ],
    }
    coverage = (
        data.groupby("season")["snapshot_gameweek"]
        .agg(["min", "max", "nunique"])
        .reset_index()
    )
    return {
        "artifact_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizon_gameweeks": horizon,
        "target": label,
        "target_kind": target_kind,
        "selected_model": selected_model,
        "feature_schema": schema.as_dict(),
        "training_rows": len(data),
        "training_cutoff": data["snapshot_timestamp"].max().isoformat(),
        "training_coverage": [
            {
                "season": str(row.season),
                "first_gameweek": int(row.min),
                "last_gameweek": int(row.max),
                "snapshot_gameweeks": int(row.nunique),
            }
            for row in coverage.itertuples(index=False)
        ],
        "source_training_tables": [
            str(path.resolve()) for path in (source_paths or [])
        ],
        "random_seed": random_seed,
        "minimum_feature_coverage": minimum_feature_coverage,
        "evaluation": evaluation,
        "probability_calibration": probability_calibration,
    }


def _fit_probability_calibrator(
    predictions: pd.DataFrame,
) -> tuple[LogisticRegression, dict[str, Any]]:
    """Fit sigmoid calibration on OOF scores and audit a later holdout slice."""
    ordered = predictions.sort_values("snapshot_timestamp").reset_index(drop=True)
    unique_times = ordered["snapshot_timestamp"].drop_duplicates().tolist()
    if len(unique_times) < 3:
        raise ValueError("Probability calibration requires at least three origins")
    split_index = max(1, int(len(unique_times) * 2 / 3))
    calibration_times = set(unique_times[:split_index])
    calibration_rows = ordered[ordered["snapshot_timestamp"].isin(calibration_times)]
    audit_rows = ordered[~ordered["snapshot_timestamp"].isin(calibration_times)]
    if calibration_rows["actual"].nunique() < 2:
        raise ValueError("Probability calibration requires both outcome classes")
    audit_calibrator = LogisticRegression(C=1.0, max_iter=1000)
    audit_calibrator.fit(
        calibration_rows[["prediction"]].to_numpy(dtype=float),
        calibration_rows["actual"].to_numpy(dtype=int),
    )
    audit_raw = np.clip(audit_rows["prediction"].to_numpy(dtype=float), 0, 1)
    audit_calibrated = audit_calibrator.predict_proba(
        audit_rows[["prediction"]].to_numpy(dtype=float)
    )[:, 1]
    calibrator = LogisticRegression(C=1.0, max_iter=1000)
    calibrator.fit(
        ordered[["prediction"]].to_numpy(dtype=float),
        ordered["actual"].to_numpy(dtype=int),
    )
    return calibrator, {
        "method": "sigmoid_logistic",
        "source": "out_of-fold temporal predictions",
        "audit_rows": len(audit_rows),
        "audit_origins": len(unique_times) - split_index,
        "audit_brier_raw": float(
            brier_score_loss(audit_rows["actual"], audit_raw)
        ),
        "audit_brier_calibrated": float(
            brier_score_loss(audit_rows["actual"], audit_calibrated)
        ),
    }


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _select_folds(
    folds: list[TemporalFold],
    maximum: int | None,
) -> list[TemporalFold]:
    if maximum is None:
        return folds
    if maximum <= 0:
        raise ValueError("max_validation_folds must be positive")
    if len(folds) <= maximum:
        return folds
    indexes = np.linspace(0, len(folds) - 1, num=maximum, dtype=int)
    return [folds[index] for index in indexes]
