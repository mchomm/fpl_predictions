"""Purged rolling-origin validation for future gameweek targets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class TemporalValidationError(ValueError):
    """Raised when data cannot support the requested temporal validation."""


@dataclass(frozen=True, slots=True)
class TemporalFold:
    """Train and validation row indexes for one prediction-time origin."""

    fold: int
    validation_season: str
    validation_gameweek: int
    train_index: pd.Index
    validation_index: pd.Index


def rolling_origin_folds(
    frame: pd.DataFrame,
    horizon: int,
    min_train_periods: int = 4,
) -> list[TemporalFold]:
    """Create folds with same-season labels purged through the validation GW."""
    required = {
        "season",
        "snapshot_gameweek",
        "snapshot_timestamp",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise TemporalValidationError(
            "Temporal validation is missing columns: "
            + ", ".join(sorted(missing))
        )
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if min_train_periods <= 0:
        raise ValueError("min_train_periods must be positive")
    if frame.empty:
        raise TemporalValidationError("Temporal validation received no rows")

    working = frame.copy()
    working["snapshot_timestamp"] = pd.to_datetime(
        working["snapshot_timestamp"], utc=True, errors="raise"
    )
    periods = (
        working[
            ["season", "snapshot_gameweek", "snapshot_timestamp"]
        ]
        .drop_duplicates()
        .sort_values(
            ["snapshot_timestamp", "season", "snapshot_gameweek"],
            ignore_index=True,
        )
    )
    folds: list[TemporalFold] = []
    for period in periods.itertuples(index=False):
        earlier = working["snapshot_timestamp"] < period.snapshot_timestamp
        prior_season = working["season"] != period.season
        label_finished = (
            working["snapshot_gameweek"] + horizon
            <= period.snapshot_gameweek
        )
        train_mask = earlier & (prior_season | label_finished)
        train_period_count = working.loc[
            train_mask, ["season", "snapshot_gameweek"]
        ].drop_duplicates().shape[0]
        if train_period_count < min_train_periods:
            continue
        validation_mask = (
            (working["season"] == period.season)
            & (
                working["snapshot_gameweek"]
                == period.snapshot_gameweek
            )
        )
        if not validation_mask.any():
            continue
        folds.append(
            TemporalFold(
                fold=len(folds) + 1,
                validation_season=str(period.season),
                validation_gameweek=int(period.snapshot_gameweek),
                train_index=working.index[train_mask],
                validation_index=working.index[validation_mask],
            )
        )

    if not folds:
        raise TemporalValidationError(
            "No rolling-origin folds are possible. Collect more labeled "
            "snapshot gameweeks or reduce min_train_periods."
        )
    return folds

