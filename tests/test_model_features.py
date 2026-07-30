"""Tests for explicit model feature contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_predictions.modelling.features import (
    ModelFeatureError,
    infer_feature_schema,
    prepare_features,
)


def test_schema_excludes_identity_timestamps_and_labels() -> None:
    frame = pd.DataFrame(
        [
            {
                "player_id": 101,
                "display_name": "Player",
                "snapshot_timestamp": "2026-08-01T00:00:00Z",
                "price": "7.5",
                "form": "4.2",
                "position_short_name": "MID",
                "club_id": 10,
                "recent_points_mean_3": 5.0,
                "label_next_1_gameweek": 8,
            }
        ]
    )

    schema = infer_feature_schema(frame)

    assert "player_id" not in schema.columns
    assert "display_name" not in schema.columns
    assert "snapshot_timestamp" not in schema.columns
    assert "label_next_1_gameweek" not in schema.columns
    assert schema.numeric == ("price", "form", "recent_points_mean_3")
    assert schema.categorical == ("position_short_name", "club_id")


def test_prediction_requires_saved_feature_schema() -> None:
    training = pd.DataFrame([{"price": 5.0, "form": 2.0}])
    schema = infer_feature_schema(training)

    with pytest.raises(ModelFeatureError, match="missing model features"):
        prepare_features(pd.DataFrame([{"price": 5.5}]), schema)


def test_sparse_optional_feature_is_not_silently_enabled() -> None:
    frame = pd.DataFrame(
        {
            "price": [5.0, 6.0, 7.0, 8.0],
            "form": [1.0, pd.NA, pd.NA, pd.NA],
        }
    )

    schema = infer_feature_schema(frame, minimum_coverage=0.5)

    assert schema.numeric == ("price",)
