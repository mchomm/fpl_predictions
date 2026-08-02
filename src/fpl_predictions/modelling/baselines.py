"""Transparent FPL-relevant point prediction baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import clone


class HistoricalMeanRegressor(RegressorMixin, BaseEstimator):
    """Predict the mean future-points label observed in training."""

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> "HistoricalMeanRegressor":
        del X
        values = np.asarray(y, dtype=float)
        if values.size == 0:
            raise ValueError("HistoricalMeanRegressor requires training rows")
        self.mean_ = float(np.mean(values))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise ValueError("HistoricalMeanRegressor has not been fitted")
        return np.full(len(X), self.mean_, dtype=float)


class RecentPointsRegressor(RegressorMixin, BaseEstimator):
    """Project a past per-gameweek points mean across the target horizon."""

    def __init__(self, recent_column: str, horizon: int) -> None:
        self.recent_column = recent_column
        self.horizon = horizon

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> "RecentPointsRegressor":
        if self.recent_column not in X:
            raise ValueError(f"Missing recent-points feature {self.recent_column!r}")
        values = np.asarray(y, dtype=float)
        if values.size == 0:
            raise ValueError("RecentPointsRegressor requires training rows")
        self.fallback_ = float(np.mean(values))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "fallback_"):
            raise ValueError("RecentPointsRegressor has not been fitted")
        recent = pd.to_numeric(X[self.recent_column], errors="coerce")
        predictions = recent * self.horizon
        return predictions.fillna(self.fallback_).to_numpy(dtype=float)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "recent_column": self.recent_column,
            "horizon": self.horizon,
        }


class RecentWithModelFallbackRegressor(RegressorMixin, BaseEstimator):
    """Use transparent recent minutes, with a learned cold-start fallback."""

    def __init__(
        self,
        recent_column: str,
        horizon: int,
        fallback_estimator: object,
    ) -> None:
        self.recent_column = recent_column
        self.horizon = horizon
        self.fallback_estimator = fallback_estimator

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> "RecentWithModelFallbackRegressor":
        if self.recent_column not in X:
            raise ValueError(f"Missing recent feature {self.recent_column!r}")
        self.fitted_fallback_ = clone(self.fallback_estimator)
        self.fitted_fallback_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "fitted_fallback_"):
            raise ValueError(
                "RecentWithModelFallbackRegressor has not been fitted"
            )
        recent = pd.to_numeric(X[self.recent_column], errors="coerce")
        predictions = recent * self.horizon
        missing = predictions.isna()
        if missing.any():
            fallback = np.asarray(self.fitted_fallback_.predict(X), dtype=float)
            predictions.loc[missing] = fallback[missing.to_numpy()]
        return predictions.to_numpy(dtype=float)
