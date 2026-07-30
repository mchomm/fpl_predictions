"""Point-forecast metrics, ranking quality, and calibration summaries."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, float | int | None]:
    """Calculate aggregate error, correlation, and within-GW ranking metrics."""
    required = {"actual", "prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            "Predictions are missing columns: " + ", ".join(sorted(missing))
        )
    valid = predictions.dropna(subset=["actual", "prediction"])
    if valid.empty:
        raise ValueError("No valid predictions are available for evaluation")
    actual = valid["actual"].astype(float)
    predicted = valid["prediction"].astype(float)
    correlation = _safe_correlation(actual, predicted, method="pearson")

    ranking_values: list[float] = []
    group_columns = [
        column
        for column in ("season", "snapshot_gameweek")
        if column in valid
    ]
    if len(group_columns) == 2:
        for _, group in valid.groupby(group_columns):
            value = _safe_correlation(
                group["actual"].astype(float),
                group["prediction"].astype(float),
                method="spearman",
            )
            if value is not None:
                ranking_values.append(value)
    else:
        value = _safe_correlation(actual, predicted, method="spearman")
        if value is not None:
            ranking_values.append(value)

    return {
        "rows": len(valid),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "correlation": correlation,
        "ranking_correlation": (
            float(np.mean(ranking_values)) if ranking_values else None
        ),
        "mean_prediction": float(predicted.mean()),
        "mean_actual": float(actual.mean()),
    }


def calibration_by_prediction_bins(
    predictions: pd.DataFrame,
    bins: int = 10,
) -> pd.DataFrame:
    """Compare average predicted and actual points in predicted-point quantiles."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    valid = predictions.dropna(subset=["actual", "prediction"]).copy()
    if valid.empty:
        raise ValueError("No valid predictions are available for calibration")
    unique_predictions = valid["prediction"].nunique()
    effective_bins = min(bins, unique_predictions)
    if effective_bins <= 1:
        valid["prediction_bin"] = "all"
    else:
        valid["prediction_bin"] = pd.qcut(
            valid["prediction"],
            q=effective_bins,
            duplicates="drop",
        ).astype(str)
    result = (
        valid.groupby("prediction_bin", observed=True, as_index=False)
        .agg(
            count=("actual", "size"),
            mean_prediction=("prediction", "mean"),
            mean_actual=("actual", "mean"),
        )
        .sort_values("mean_prediction", ignore_index=True)
    )
    result["calibration_error"] = (
        result["mean_prediction"] - result["mean_actual"]
    )
    return result


def _safe_correlation(
    first: pd.Series,
    second: pd.Series,
    method: str,
) -> float | None:
    if len(first) < 2 or first.nunique() <= 1 or second.nunique() <= 1:
        return None
    value = first.corr(second, method=method)
    if pd.isna(value):
        return None
    return float(value)

