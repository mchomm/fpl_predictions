"""Candidate model construction."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fpl_predictions.modelling.baselines import (
    HistoricalMeanRegressor,
    RecentPointsRegressor,
    RecentWithModelFallbackRegressor,
)
from fpl_predictions.modelling.features import FeatureSchema


TEAM_STRENGTH_FEATURE_PREFIXES = (
    "club_attack_strength",
    "club_goals_conceded_strength",
    "club_strength_matches",
    "upcoming_opponent_",
    "upcoming_attacking_fixture_",
    "upcoming_defensive_fixture_",
)


def candidate_models(
    schema: FeatureSchema,
    horizon: int,
    random_seed: int = 42,
    target_kind: str = "points",
) -> dict[str, object]:
    """Return deterministic baseline and supervised candidate estimators."""
    models: dict[str, object] = {
        "historical_mean": HistoricalMeanRegressor(),
    }
    recent_prefixes = {
        "points": "recent_points_mean_",
        "minutes": "recent_minutes_mean_",
        "appearances": "recent_appearance_rate_",
        "starts": "recent_start_rate_",
    }
    recent_prefix = recent_prefixes[target_kind]
    recent_columns = [
        column for column in schema.numeric if column.startswith(recent_prefix)
    ]
    if recent_columns:
        models[f"recent_{target_kind}_mean"] = RecentPointsRegressor(
            recent_columns[0], horizon
        )

    linear_preprocessor = _preprocessor(schema, scale_numeric=True)
    models["ridge"] = Pipeline(
        [
            ("preprocess", linear_preprocessor),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    forest_preprocessor = _preprocessor(schema, scale_numeric=False)
    models["random_forest"] = Pipeline(
        [
            ("preprocess", forest_preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=4,
                    random_state=random_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    if target_kind in {"minutes", "appearances", "starts"} and recent_columns:
        models[f"recent_{target_kind}_with_model_fallback"] = (
            RecentWithModelFallbackRegressor(
                recent_columns[0],
                horizon,
                models["random_forest"],
            )
        )
    base_schema = _without_team_strength(schema)
    if base_schema != schema:
        models["random_forest_without_team_strength"] = Pipeline(
            [
                (
                    "preprocess",
                    _preprocessor(base_schema, scale_numeric=False),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_leaf=4,
                        random_state=random_seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return models


def _without_team_strength(schema: FeatureSchema) -> FeatureSchema:
    """Return the pre-upgrade feature set for an honest validation ablation."""
    return FeatureSchema(
        numeric=tuple(
            column
            for column in schema.numeric
            if not column.startswith(TEAM_STRENGTH_FEATURE_PREFIXES)
        ),
        categorical=schema.categorical,
    )


def _preprocessor(
    schema: FeatureSchema,
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        (
            "impute",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
                add_indicator=True,
            ),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    transformers: list[tuple[str, object, list[str]]] = [
        ("numeric", Pipeline(numeric_steps), list(schema.numeric))
    ]
    if schema.categorical:
        categorical = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical, list(schema.categorical))
        )
    return ColumnTransformer(transformers, remainder="drop")
