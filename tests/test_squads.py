"""Unit tests for Phase 4 squad validation, projection, and ratings."""

from __future__ import annotations

from dataclasses import replace
import json

import pandas as pd
import pytest

from fpl_predictions.squads.generation import generate_reference_population
from fpl_predictions.squads.optimizer import (
    SquadOptimizationError,
    optimize_squad,
)
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
from fpl_predictions.squads.simulation import simulate_selection
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
            "appearance_probability_1": [1.0] * 15,
            "start_probability_1": [1.0] * 15,
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


def test_budget_can_be_ignored_for_editing_but_not_for_rating(
    rules: SquadRules,
    players: pd.DataFrame,
    selection: SquadSelection,
) -> None:
    expensive_players = players.copy()
    expensive_players["price"] = expensive_players["price"] + 2.0
    selection_with_high_limit = replace(selection, budget_limit=200.0)

    with pytest.raises(SquadValidationError, match="exceeds budget 100.0"):
        validate_squad(selection_with_high_limit, expensive_players, rules)

    editable = validate_squad(
        selection_with_high_limit,
        expensive_players,
        rules,
        enforce_budget=False,
    )
    assert editable.total_cost == 115.0
    assert editable.budget_limit == 100.0


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


def test_simulation_matches_raw_projection_when_everyone_appears(
    rules: SquadRules,
    players: pd.DataFrame,
    selection: SquadSelection,
    predictions: pd.DataFrame,
) -> None:
    healthy = players.assign(
        status="a",
        chance_of_playing_this_round=None,
        chance_of_playing_next_round=None,
    )
    merged = healthy.merge(predictions, on="player_id", validate="one_to_one")
    result = simulate_selection(
        merged, selection, rules, horizon=3, simulations=20, seed=7
    )

    assert result.projection.starting_xi_points == pytest.approx(186)
    assert result.projection.captain_points == pytest.approx(30)
    assert result.projection.overall_points == pytest.approx(216)
    assert result.expected_autosub_points == 0
    assert result.vice_captain_takeover_probability == 0


def test_simulation_applies_legal_autosub_and_vice_takeover(
    rules: SquadRules,
    players: pd.DataFrame,
    selection: SquadSelection,
    predictions: pd.DataFrame,
) -> None:
    healthy = players.assign(
        status="a",
        chance_of_playing_this_round=None,
        chance_of_playing_next_round=None,
    )
    merged = healthy.merge(predictions, on="player_id", validate="one_to_one")
    # Starting DEF 3 and captain 15 never appear. Bench DEF 6 and vice 14 do.
    merged.loc[merged["player_id"].isin([3, 15]), "appearance_probability_1"] = 0
    for horizon in (1, 3):
        merged.loc[
            merged["player_id"].isin([3, 15]),
            f"predicted_points_{horizon}",
        ] = 0
    result = simulate_selection(
        merged, selection, rules, horizon=3, simulations=20, seed=11
    )

    assert result.expected_autosub_points == pytest.approx(26)
    assert result.autosub_probability == 1
    assert result.expected_vice_captain_points == pytest.approx(28)
    assert result.vice_captain_takeover_probability == 1


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


def test_reference_population_uses_same_availability_simulation_policy(
    rules: SquadRules,
    players: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    references = generate_reference_population(
        players,
        predictions,
        rules,
        horizon=3,
        size=2,
        strategy="human_like",
        seed=17,
        target_cost=85.0,
        budget_band=0.0,
        availability_simulations=10,
    )

    assert references["simulation_count"].eq(10).all()
    assert references["expected_autosub_points"].ge(0).all()
    assert references["vice_captain_takeover_probability"].between(0, 1).all()


def test_optimizer_produces_exact_legal_improvement_with_transfer_limit(
    rules: SquadRules,
    players: pd.DataFrame,
    selection: SquadSelection,
    predictions: pd.DataFrame,
) -> None:
    alternative = players.iloc[[2]].copy()
    alternative["player_id"] = 16
    alternative["display_name"] = "Elite Defender"
    alternative["club_id"] = 1
    expanded_players = pd.concat([players, alternative], ignore_index=True)
    alternative_prediction = predictions.iloc[[2]].copy()
    alternative_prediction["player_id"] = 16
    alternative_prediction["predicted_points_1"] = 20.0
    alternative_prediction["predicted_points_3"] = 50.0
    expanded_predictions = pd.concat(
        [predictions, alternative_prediction], ignore_index=True
    )

    result = optimize_squad(
        expanded_players,
        expanded_predictions,
        rules,
        horizon=3,
        current_squad=selection,
        max_transfers=1,
    )

    validate_squad(result.selection, expanded_players, rules)
    assert result.transfers_in == (16,)
    assert len(result.transfers_out) == 1
    assert result.projected_points_gain is not None
    assert result.projected_points_gain > 0
    assert result.selection.captain == 16


def test_optimizer_reports_infeasible_locked_club_limit(
    rules: SquadRules,
    players: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    alternatives = pd.concat([players, players.iloc[[0]]], ignore_index=True)
    alternatives.loc[alternatives.index[-1], "player_id"] = 16
    alternatives.loc[alternatives.index[-1], "club_id"] = 1
    extra_prediction = predictions.iloc[[0]].assign(player_id=16)
    expanded_predictions = pd.concat(
        [predictions, extra_prediction], ignore_index=True
    )

    with pytest.raises(SquadOptimizationError, match="No legal optimized squad"):
        optimize_squad(
            alternatives,
            expanded_predictions,
            rules,
            locked_player_ids=(1, 6, 11, 16),
        )


def test_optimizer_charges_only_transfers_beyond_free_allowance(
    rules: SquadRules,
    players: pd.DataFrame,
    selection: SquadSelection,
    predictions: pd.DataFrame,
) -> None:
    alternatives = players.iloc[[2, 3]].copy()
    alternatives["player_id"] = [16, 17]
    alternatives["club_id"] = [3, 4]
    alternatives["price"] = [5.0, 5.0]
    expanded_players = pd.concat([players, alternatives], ignore_index=True)
    extra_predictions = predictions.iloc[[2, 3]].copy()
    extra_predictions["player_id"] = [16, 17]
    extra_predictions["predicted_points_1"] = [20.0, 19.0]
    extra_predictions["predicted_points_3"] = [50.0, 48.0]
    expanded_predictions = pd.concat(
        [predictions, extra_predictions], ignore_index=True
    )

    result = optimize_squad(
        expanded_players,
        expanded_predictions,
        rules,
        horizon=3,
        current_squad=selection,
        max_transfers=2,
        excluded_player_ids=(3, 4),
        free_transfers=1,
        hit_cost=4.0,
    )

    assert len(result.transfers_in) == 2
    assert result.free_transfers == 1
    assert result.paid_transfers == 1
    assert result.transfer_hit_points == 4.0
    assert result.net_projected_points_gain == pytest.approx(
        result.projected_points_gain - 4.0
    )


def test_optimizer_uses_manager_sell_value_for_affordability(
    rules: SquadRules,
    players: pd.DataFrame,
    selection: SquadSelection,
    predictions: pd.DataFrame,
) -> None:
    current = SquadSelection(
        player_ids=selection.player_ids,
        starting_xi=selection.starting_xi,
        bench=selection.bench,
        captain=selection.captain,
        vice_captain=selection.vice_captain,
        source="manager_api",
        bank=0.0,
        selling_prices={
            player_id: (
                6.0 if player_id == 15 else float(
                    players.set_index("player_id").loc[player_id, "price"]
                )
            )
            for player_id in selection.player_ids
        },
    )
    alternative = players.iloc[[14]].copy()
    alternative["player_id"] = 16
    alternative["club_id"] = 1
    alternative["price"] = 7.0
    expanded_players = pd.concat([players, alternative], ignore_index=True)
    extra_prediction = predictions.iloc[[14]].assign(
        player_id=16,
        predicted_points_1=50.0,
        predicted_points_3=100.0,
    )
    expanded_predictions = pd.concat(
        [predictions, extra_prediction], ignore_index=True
    )

    with pytest.raises(SquadOptimizationError, match="No legal optimized squad"):
        optimize_squad(
            expanded_players,
            expanded_predictions,
            rules,
            horizon=3,
            current_squad=current,
            max_transfers=1,
            locked_player_ids=set(selection.player_ids) - {15},
            excluded_player_ids=(15,),
        )


def test_squad_schema_parses_json_price_maps(selection: SquadSelection) -> None:
    payload = selection.as_dict()
    payload["purchase_prices"] = {"1": 4.5}
    payload["selling_prices"] = {"1": 4.7}

    parsed = SquadSelection.from_mapping(payload)

    assert parsed.purchase_prices == {1: 4.5}
    assert parsed.selling_prices == {1: 4.7}


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
