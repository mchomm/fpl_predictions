"""Tests for API-to-table normalization."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_predictions.data.normalize import (
    NormalizationError,
    normalize_current_data,
)


def test_player_mappings_price_and_statistics(
    bootstrap_payload: dict,
    fixtures_payload: list[dict],
) -> None:
    tables = normalize_current_data(bootstrap_payload, fixtures_payload)
    players = tables["players"].set_index("player_id")

    assert players.loc[101, "price"] == pytest.approx(4.5)
    assert players.loc[101, "full_name"] == "Ada Keeper"
    assert players.loc[101, "club_name"] == "North Town"
    assert players.loc[101, "club_short_name"] == "NOR"
    assert players.loc[101, "position"] == "Goalkeeper"
    assert players.loc[101, "position_short_name"] == "GKP"
    assert players.loc[101, "ownership_percent"] == pytest.approx(12.3)
    assert players.loc[101, "total_points"] == 42
    assert players.loc[101, "photo_url"].endswith("/p50101.png")


def test_all_current_tables_are_normalized(
    bootstrap_payload: dict,
    fixtures_payload: list[dict],
) -> None:
    tables = normalize_current_data(bootstrap_payload, fixtures_payload)

    assert set(tables) == {
        "players",
        "clubs",
        "positions",
        "gameweeks",
        "fixtures",
    }
    assert tables["clubs"]["club_name"].tolist() == ["North Town", "South City"]
    assert tables["positions"]["position_short_name"].tolist() == [
        "GKP",
        "DEF",
        "MID",
        "FWD",
    ]
    assert tables["gameweeks"].loc[0, "gameweek_name"] == "Gameweek 1"
    fixture = tables["fixtures"].iloc[0]
    assert fixture["home_club_name"] == "North Town"
    assert fixture["away_club_name"] == "South City"
    assert pd.notna(fixture["gameweek_id"])


def test_unknown_club_is_not_silently_invented(
    bootstrap_payload: dict,
    fixtures_payload: list[dict],
) -> None:
    bootstrap_payload["elements"][0]["team"] = 999

    with pytest.raises(NormalizationError, match=r"Unknown club IDs: \[999\]"):
        normalize_current_data(bootstrap_payload, fixtures_payload)


def test_non_numeric_price_is_rejected(
    bootstrap_payload: dict,
    fixtures_payload: list[dict],
) -> None:
    bootstrap_payload["elements"][0]["now_cost"] = "unknown"

    with pytest.raises(NormalizationError, match="Non-numeric player prices"):
        normalize_current_data(bootstrap_payload, fixtures_payload)

