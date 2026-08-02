"""Tests for the browser-facing manual squad builder helpers."""

from __future__ import annotations

import pandas as pd

from fpl_predictions.serving.presentation import (
    build_manual_selection,
    legal_formations,
    selection_defaults,
    slot_positions,
)
from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.schemas import SquadSelection
from fpl_predictions.squads.validation import validate_squad


def _rules() -> SquadRules:
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
                {"id": 1, "singular_name": "Goalkeeper", "singular_name_short": "GKP", "squad_select": 2, "squad_min_play": 1, "squad_max_play": 1},
                {"id": 2, "singular_name": "Defender", "singular_name_short": "DEF", "squad_select": 5, "squad_min_play": 3, "squad_max_play": 5},
                {"id": 3, "singular_name": "Midfielder", "singular_name_short": "MID", "squad_select": 5, "squad_min_play": 2, "squad_max_play": 5},
                {"id": 4, "singular_name": "Forward", "singular_name_short": "FWD", "squad_select": 3, "squad_min_play": 1, "squad_max_play": 3},
            ],
        }
    )


def _players() -> pd.DataFrame:
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    position_ids = [1] * 2 + [2] * 5 + [3] * 5 + [4] * 3
    return pd.DataFrame(
        {
            "player_id": range(1, 16),
            "display_name": [f"Player {value}" for value in range(1, 16)],
            "club_id": [(value - 1) % 5 + 1 for value in range(1, 16)],
            "club_name": [f"Club {(value - 1) % 5 + 1}" for value in range(1, 16)],
            "club_short_name": [f"C{(value - 1) % 5 + 1}" for value in range(1, 16)],
            "position_id": position_ids,
            "position_short_name": positions,
            "price": [4.0] * 15,
        }
    )


def _selection() -> SquadSelection:
    return SquadSelection(
        player_ids=tuple(range(1, 16)),
        starting_xi=(1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15),
        bench=(2, 6, 7, 12),
        captain=15,
        vice_captain=14,
    )


def test_manual_builder_exposes_every_legal_formation() -> None:
    rules = _rules()

    assert legal_formations(rules) == (
        "3-4-3",
        "3-5-2",
        "4-3-3",
        "4-4-2",
        "4-5-1",
        "5-2-3",
        "5-3-2",
        "5-4-1",
    )
    for formation in legal_formations(rules):
        starters, bench = slot_positions(formation, rules)
        assert len(starters) == 11
        assert len(bench) == 4
        assert starters.count("GKP") == bench.count("GKP") == 1


def test_existing_squad_can_be_reshaped_without_losing_players() -> None:
    rules = _rules()
    starters, bench = selection_defaults(
        _selection(), "4-4-2", _players(), rules
    )

    assert len(starters) == 11
    assert len(bench) == 4
    assert set(starters + bench) == set(range(1, 16))
    position_lookup = _players().set_index("player_id")["position_short_name"]
    assert sum(position_lookup.loc[player] == "DEF" for player in starters) == 4
    assert sum(position_lookup.loc[player] == "FWD" for player in starters) == 2


def test_complete_manual_slots_build_a_valid_budgeted_squad() -> None:
    selection = build_manual_selection(
        (1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15),
        (2, 6, 7, 12),
        15,
        14,
        bank=1.5,
        budget_limit=100.0,
    )

    validated = validate_squad(selection, _players(), _rules())

    assert selection.source == "manual"
    assert selection.bank == 1.5
    assert validated.formation == "3-4-3"
