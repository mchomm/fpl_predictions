"""Tests for global, constraint-based lineup-role reconciliation."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_predictions.modelling.lineup_roles import (
    reconcile_lineup_probabilities,
)


def test_reconciliation_applies_joint_constraints_without_name_rules() -> None:
    frame = pd.DataFrame(
        {
            "player_id": range(1, 17),
            "club_id": [1] * 16,
            "position_short_name": ["GKP"] * 3 + ["DEF"] * 6 + ["MID"] * 4 + ["FWD"] * 3,
            "appearance_probability_1": [0.9, 0.6, 0.2] + [0.8] * 13,
            "start_probability_1": [0.7, 0.2, 0.1] + [0.4] * 13,
        }
    )

    result, audit = reconcile_lineup_probabilities(frame)

    goalkeepers = result.loc[result["position_short_name"].eq("GKP")]
    outfield = result.loc[~result["position_short_name"].eq("GKP")]
    assert goalkeepers["start_probability_1"].sum() == pytest.approx(1.0)
    assert outfield["start_probability_1"].sum() == pytest.approx(10.0)
    assert result["start_probability_1"].le(
        result["appearance_probability_1"]
    ).all()
    assert goalkeepers.sort_values("independent_start_probability_1")[
        "start_probability_1"
    ].is_monotonic_increasing
    assert audit["player_or_club_overrides"] == 0


def test_reconciliation_rejects_invalid_probabilities() -> None:
    frame = pd.DataFrame(
        {
            "player_id": [1],
            "club_id": [1],
            "position_short_name": ["GKP"],
            "appearance_probability_1": [1.2],
            "start_probability_1": [0.5],
        }
    )

    with pytest.raises(ValueError, match="between zero and one"):
        reconcile_lineup_probabilities(frame)
