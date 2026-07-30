"""Unit tests for Phase 4 squad validation, projection, and ratings."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_predictions.squads.generation import generate_reference_population
from fpl_predictions.squads.projection import project_squad
from fpl_predictions.squads.ratings import midrank_percentile, rate_squad
from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.schemas import SquadSelection
from fpl_predictions.squads.validation import (
    SquadValidationError,
    validate_squad,
)


@pytest.fixture
def rules() -> SquadRules:
    return SquadRules.from_bootstrap(
        {
            "game_settings": {
                "squad_squadsize": 15,
                "squad_squadplay": 11,
                "squad_team_limit": 3,
                "squad_total_spend": 1000,
                "ui_currency_multiplier": 10,
            },
            "element_types": [
                {
                    "id": 1,
                    "singular_name": "Goalkeeper",
                    "singular_name_short": "GKP",
                    "squad_select": 2,
                    "squad_min_play": 1,
                    "squad_max_play": 1,
                },
                {
                    "id": 2,
                    "singular_name": "Defender",
                    "singular_name_short": "DEF",
                    "squad_select": 5,
                    "squad_min_play": 3,
                    "squad_max_play": 5,
                },
                {
                    "id": 3,
                    "singular_name": "Midfielder",
                    "singular_name_short": "MID",
                    "squad_select": 5,
                    "squad_min_play": 2,
                    "squad_max_play": 5,
                },
                {
                    "id": 4,
                    "singular_name": "Forward",
                    "singular_name_short": "FWD",
                    "squad_select": 3,
                    "squad_min_play": 1,
                    "squad_max_play": 3,
                },
            ],
        }
    )


@pytest.fixture
def players() -> pd.DataFrame:
    position_ids = [1] * 2 + [2] * 5 + [3] * 5 + [4] * 3
    short_names = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    prices = [4.5] * 2 + [5.0] * 5 + [6.0] * 5 + [7.0] * 3
    return pd.DataFrame(
        {
            "player_id": range(1, 16),
            "display_name": [f"Player {value}" for value in range(1, 16)],
            "club_id": [(value - 1) % 5 + 1 for value in range(1, 16)],
            "position_id": position_ids,
            "position_short_name": short_names,
            "price": prices,
            "ownership_percent": range(1, 16),
            "status": ["a"] * 14 + ["d"],
            "chance_of_playing_this_round": [None] * 14 + [50],
            "chance_of_playing_next_round": [None] * 15,
        }
    )


@pytest.fixture
def selection() -> SquadSelection:
    return SquadSelection(
        player_ids=tuple(range(1, 16)),
        starting_xi=(1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15),
        bench=(2, 6, 7, 12),
        captain=15,
        vice_captain=14,
    )


@pytest.fixture
def predictions() -> pd.DataFrame:
    ids = pd.Series(range(1, 16))
    return pd.DataFrame(
        {
            "player_id": ids,
            "predicted_points_1": ids.astype(float),
            "predicted_points_3": ids.astype(float) * 2,
        }
    )


def test_rules_are_derived_from_api(rules: SquadRules) -> None:
    assert rules.squad_size == 15
    assert rules.starting_size == 11
    assert rules.budget == 100.0
    assert rules.position_by_short_name["DEF"].squad_count == 5


def test_valid_squad_has_formation_and_cost(
    rules: SquadRules,
    players: pd.DataFrame,
    selection: SquadSelection,
) -> None:
    validated = validate_squad(selection, players, rules)
    assert validated.formation == "3-4-3"
    assert validated.total_cost == 85.0


def test_validation_reports_duplicate_and_bad_captain(
    rules: SquadRules,
    players: pd.DataFrame,
    selection: SquadSelection,
) -> None:
    invalid = SquadSelection(
        player_ids=selection.player_ids[:-1] + (14,),
        starting_xi=selection.starting_xi,
        bench=selection.bench,
        captain=2,
        vice_captain=14,
    )
    with pytest.raises(SquadValidationError) as error:
        validate_squad(invalid, players, rules)
    assert "duplicate" in str(error.value)
    assert "captain must be in the starting XI" in str(error.value)


def test_projection_aggregates_positions_bench_and_captain(
    rules: SquadRules,
    players: pd.DataFrame,
    selection: SquadSelection,
    predictions: pd.DataFrame,
) -> None:
    validated = validate_squad(selection, players, rules)
    projection = project_squad(validated, predictions, rules, horizon=3)
    assert projection.goalkeeper_points == 2.0
    assert projection.defence_points == 24.0
    assert projection.midfield_points == 76.0
    assert projection.forward_points == 76.5
    assert projection.bench_points == 54.0
    assert projection.captain_points == 22.5
    assert projection.overall_points == 201.0
    assert "availability-adjusted" in projection.warnings[0]


def test_midrank_percentile_and_direct_overall_rating() -> None:
    from fpl_predictions.squads.projection import SquadProjection

    projection = SquadProjection(
        horizon=1,
        formation="3-4-3",
        goalkeeper_points=2,
        defence_points=2,
        midfield_points=2,
        forward_points=2,
        bench_points=2,
        captain_points=2,
        starting_xi_points=8,
        overall_points=10,
        warnings=(),
    )
    references = pd.DataFrame(
        {
            "goalkeeper_points": [1, 2, 2, 4],
            "defence_points": [1, 2, 2, 4],
            "midfield_points": [1, 2, 2, 4],
            "forward_points": [1, 2, 2, 4],
            "bench_points": [1, 2, 2, 4],
            "captain_points": [1, 2, 2, 4],
            "overall_points": [5, 9, 11, 12],
        }
    )
    assert midrank_percentile(2, references["defence_points"]) == 50.0
    rating = rate_squad(projection, references, "test population")
    assert rating.scores["defence"] == 50.0
    assert rating.scores["overall"] == 50.0
    assert "not an average" in rating.interpretation


def test_reference_generation_is_deterministic_and_legal(
    rules: SquadRules,
    players: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    first = generate_reference_population(
        players,
        predictions,
        rules,
        horizon=3,
        size=4,
        strategy="price_aware",
        seed=7,
    )
    second = generate_reference_population(
        players,
        predictions,
        rules,
        horizon=3,
        size=4,
        strategy="price_aware",
        seed=7,
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 4
    assert first["total_cost"].le(rules.budget).all()
    assert first["formation"].eq("3-4-3").all()
