"""Tests for local OCR matching and real screenshot reconstruction."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pandas as pd
import pytest

from fpl_predictions.screenshot.recognition import (
    OCRSlot,
    _assign_players,
    _text_similarity,
    recognize_screenshot,
)
from fpl_predictions.squads.rules import SquadRules


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


def test_text_similarity_handles_accents_noise_and_truncation() -> None:
    assert _text_similarity("F.Kadioglu |", "F.Kadıoğlu") > 0.9
    assert _text_similarity("D calvert-Le... J", "Calvert-Lewin") > 0.8
    assert _text_similarity("Grok", "Groß") > 0.9


def test_assignment_uses_positions_and_squad_rules_for_ambiguous_names() -> None:
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    players = pd.DataFrame(
        {
            "player_id": range(1, 16),
            "display_name": ["Hughes" if value in (8, 3) else f"Player {value}" for value in range(1, 16)],
            "full_name": [f"Player {value}" for value in range(1, 16)],
            "surname": ["Hughes" if value in (8, 3) else str(value) for value in range(1, 16)],
            "club_id": [(value - 1) % 5 + 1 for value in range(1, 16)],
            "position_short_name": positions,
            "price": [4.0] * 15,
            "status": ["a"] * 15,
        }
    )
    starter_positions = ["GKP"] + ["DEF"] * 3 + ["MID"] * 4 + ["FWD"] * 3
    starter_ids = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15]
    slots = []
    for index, (position, player_id) in enumerate(zip(starter_positions, starter_ids, strict=True)):
        reading = "Hughes" if player_id == 8 else f"Player {player_id}"
        slots.append(OCRSlot(f"starter:{index}", "starter", position, 0.5, 0.5, (reading,)))
    for index, player_id in enumerate((2, 6, 7, 12)):
        slots.append(OCRSlot(f"bench:{index + 1}", "bench", None, 0.5, 0.9, (f"Player {player_id}",)))

    result = _assign_players(tuple(slots), players, _rules(), {})

    assigned = {row["slot_id"]: row for row in result["assigned"]}
    assert assigned["starter:4"]["player_id"] == 8
    assert assigned["starter:4"]["position"] == "MID"


@pytest.mark.skipif(
    not Path("examples/max_palmer_version.jpeg").is_file(),
    reason="real screenshot fixture is not available",
)
def test_real_max_palmer_screenshot_matches_annotation() -> None:
    command = os.environ.get("FPL_TEST_TESSERACT") or shutil.which("tesseract")
    if not command:
        pytest.skip("Tesseract is not installed")
    players = pd.read_parquet(
        "data/processed/20260730T162356213385Z/players.parquet"
    )
    result = recognize_screenshot(
        Path("examples/max_palmer_version.jpeg"),
        players,
        _rules(),
        tesseract_command=command,
    )
    expected = json.loads(
        Path("examples/screenshot-squad-max-palmer-2026-27-gw1.json").read_text()
    )

    assert list(result.selection.player_ids) == expected["player_ids"]
    assert list(result.selection.starting_xi) == expected["starting_xi"]
    assert list(result.selection.bench) == expected["bench"]
    assert result.selection.captain == expected["captain"]
    assert result.selection.vice_captain == expected["vice_captain"]
    assert result.audit["formation"] == "3-4-3"
    assert result.audit["review_required"] is False
