"""Explicit model feature contracts and input preparation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

NUMERIC_FEATURE_CANDIDATES = (
    "price",
    "ownership_percent",
    "chance_of_playing_next_round",
    "chance_of_playing_this_round",
    "form",
    "points_per_game",
    "total_points",
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "value_form",
    "value_season",
    "transfers_in_event",
    "transfers_out_event",
)

CATEGORICAL_FEATURE_CANDIDATES = (
    "position_short_name",
    "status",
    "club_name",
    "club_id",
)

DERIVED_NUMERIC_PREFIXES = (
    "recent_points_",
    "recent_minutes_",
    "recent_gameweeks_",
    "upcoming_fixture_",
    "upcoming_home_fixture_",
    "upcoming_away_fixture_",
)


class ModelFeatureError(ValueError):
    """Raised when a table cannot satisfy a model feature contract."""


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """Ordered columns and types used by a fitted model."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureSchema":
        return cls(
            numeric=tuple(value["numeric"]),
            categorical=tuple(value["categorical"]),
        )


def infer_feature_schema(
    frame: pd.DataFrame,
    minimum_coverage: float = 0.5,
) -> FeatureSchema:
    """Select approved features with sufficient non-null training coverage."""
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in the interval (0, 1]")
    numeric_candidates = [
        column for column in NUMERIC_FEATURE_CANDIDATES if column in frame.columns
    ]
    numeric_candidates.extend(
        column
        for column in frame.columns
        if column.startswith(DERIVED_NUMERIC_PREFIXES)
        and column not in numeric_candidates
    )
    numeric = [
        column
        for column in numeric_candidates
        if pd.to_numeric(frame[column], errors="coerce").notna().mean()
        >= minimum_coverage
    ]
    categorical_candidates = [
        column
        for column in CATEGORICAL_FEATURE_CANDIDATES
        if column in frame.columns
    ]
    categorical = [
        column
        for column in categorical_candidates
        if frame[column].notna().mean() >= minimum_coverage
    ]
    if not numeric:
        raise ModelFeatureError(
            "No approved numeric model features were found in the training table"
        )
    return FeatureSchema(tuple(numeric), tuple(categorical))


def prepare_features(
    frame: pd.DataFrame,
    schema: FeatureSchema,
) -> pd.DataFrame:
    """Validate, order, and coerce a frame for training or prediction."""
    missing = set(schema.columns).difference(frame.columns)
    if missing:
        raise ModelFeatureError(
            "Prediction data is missing model features: "
            + ", ".join(sorted(missing))
        )
    result = frame.loc[:, schema.columns].copy()
    for column in schema.numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in schema.categorical:
        result[column] = result[column].astype("string").fillna("unknown")
    return result
