"""Tests for model metrics and calibration."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_predictions.modelling.evaluation import (
    calibration_by_prediction_bins,
    prediction_metrics,
)


def test_prediction_metrics_include_error_and_ranking() -> None:
    predictions = pd.DataFrame(
        {
            "season": ["2026-27"] * 3,
            "snapshot_gameweek": [4] * 3,
            "actual": [1.0, 2.0, 3.0],
            "prediction": [1.0, 2.0, 4.0],
        }
    )

    metrics = prediction_metrics(predictions)

    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx((1 / 3) ** 0.5)
    assert metrics["correlation"] is not None
    assert metrics["ranking_correlation"] == pytest.approx(1.0)


def test_calibration_bins_report_predicted_vs_actual() -> None:
    predictions = pd.DataFrame(
        {
            "actual": [1.0, 2.0, 8.0, 10.0],
            "prediction": [2.0, 3.0, 7.0, 9.0],
        }
    )

    calibration = calibration_by_prediction_bins(predictions, bins=2)

    assert calibration["count"].sum() == 4
    assert set(calibration.columns) == {
        "prediction_bin",
        "count",
        "mean_prediction",
        "mean_actual",
        "calibration_error",
    }

