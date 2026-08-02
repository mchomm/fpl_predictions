"""Unit tests for Phase 4 squad validation, projection, and ratings."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fpl_predictions.squads.generation import generate_reference_population
from fpl_predictions.squads.projection import (
    player_projection_details,
    project_squad,
)
from fpl_predictions.squads.ratings import (
    midrank_percentile,
    rate_squad,
    school_style_score,
)
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
            "predicted_minutes_1": ids.astype(float) * 4,
            "predicted_minutes_3": ids.astype(float) * 12,
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


def test_validation_rejects_unsupported_active_chip(
    rules: SquadRules,
    players: pd.DataFrame,
    selection: SquadSelection,
) -> None:
    chipped = SquadSelection(
        player_ids=selection.player_ids,
        starting_xi=selection.starting_xi,
        bench=selection.bench,
        captain=selection.captain,
        vice_captain=selection.vice_captain,
        active_chip="bboost",
    )
    with pytest.raises(SquadValidationError, match="active_chip"):
        validate_squad(chipped, players, rules)


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


def test_player_details_report_expected_minutes_without_rescaling_points(
    rules: SquadRules,
    players: pd.DataFrame,
    selection: SquadSelection,
    predictions: pd.DataFrame,
) -> None:
    validated = validate_squad(selection, players, rules)
    details = player_projection_details(validated, predictions, horizon=3)
    captain = next(item for item in details if item["is_captain"])

    assert len(details) == 15
    assert captain["player_id"] == 15
    assert captain["predicted_points"] == 30
    assert captain["expected_minutes"] == 180
    assert captain["availability_adjusted_expected_minutes"] == 150


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
    assert rating.percentiles["defence"] == 50.0
    assert rating.percentiles["overall"] == 50.0
    assert rating.scores["defence"] == 75.0
    assert rating.scores["overall"] == 75.0
    assert "not an average" in rating.interpretation


def test_school_style_score_has_natural_anchors_and_no_perfect_score() -> None:
    population = pd.Series(range(10_000), dtype=float)
    assert school_style_score(5_000, population) == pytest.approx(75.0, abs=0.01)
    assert school_style_score(9_700, population) == pytest.approx(90.05, abs=0.1)
    assert school_style_score(9_940, population) == pytest.approx(95.0, abs=0.1)
    assert school_style_score(20_000, population) == 99.9


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
    row = first.iloc[0]
    generated = SquadSelection(
        player_ids=tuple(json.loads(row.player_ids)),
        starting_xi=tuple(json.loads(row.starting_xi)),
        bench=tuple(json.loads(row.bench)),
        captain=int(row.captain),
        vice_captain=int(row.vice_captain),
        source="generated",
    )
    validate_squad(generated, players, rules)


def test_human_like_references_are_ownership_and_budget_conditioned(
    rules: SquadRules,
    players: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    references = generate_reference_population(
        players,
        predictions,
        rules,
        horizon=3,
        size=3,
        strategy="human_like",
        seed=11,
        target_cost=85.0,
        budget_band=0.0,
    )
    assert references["total_cost"].eq(85.0).all()
    assert references["target_cost"].eq(85.0).all()
    assert references["minimum_reference_cost"].eq(85.0).all()


def test_human_like_references_require_ownership(
    rules: SquadRules,
    players: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="ownership_percent"):
        generate_reference_population(
            players.drop(columns="ownership_percent"),
            predictions,
            rules,
            horizon=3,
            size=1,
            strategy="human_like",
            target_cost=85.0,
            budget_band=0.0,
        )
