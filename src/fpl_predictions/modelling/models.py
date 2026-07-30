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
)
from fpl_predictions.modelling.features import FeatureSchema


def candidate_models(
    schema: FeatureSchema,
    horizon: int,
    random_seed: int = 42,
) -> dict[str, object]:
    """Return deterministic baseline and supervised candidate estimators."""
    models: dict[str, object] = {
        "historical_mean": HistoricalMeanRegressor(),
    }
    recent_columns = [
        column for column in schema.numeric if column.startswith("recent_points_mean_")
    ]
    if recent_columns:
        models["recent_points_mean"] = RecentPointsRegressor(
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
    return models


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
