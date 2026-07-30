"""Normalize live FPL API responses into analysis-ready tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

PHOTO_URL_TEMPLATE = (
    "https://resources.premierleague.com/premierleague/"
    "photos/players/110x140/p{player_code}.png"
)

PLAYER_REQUIRED_FIELDS = {
    "id",
    "code",
    "first_name",
    "second_name",
    "web_name",
    "team",
    "element_type",
    "now_cost",
}

PLAYER_RENAMES = {
    "id": "player_id",
    "code": "player_code",
    "second_name": "surname",
    "web_name": "display_name",
    "team": "club_id",
    "element_type": "position_id",
    "now_cost": "price",
    "photo": "photo_reference",
    "selected_by_percent": "ownership_percent",
}

PLAYER_FRONT_COLUMNS = [
    "player_id",
    "player_code",
    "first_name",
    "surname",
    "full_name",
    "display_name",
    "club_id",
    "club_name",
    "club_short_name",
    "position_id",
    "position",
    "position_short_name",
    "price",
    "photo_reference",
    "photo_url",
    "ownership_percent",
    "status",
    "chance_of_playing_next_round",
    "chance_of_playing_this_round",
    "news",
    "news_added",
]


class NormalizationError(ValueError):
    """Raised when API data cannot be normalized without inventing values."""


def normalize_current_data(
    bootstrap: Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Normalize current API payloads into five linked DataFrames."""
    clubs = normalize_clubs(bootstrap["teams"])
    positions = normalize_positions(bootstrap["element_types"])
    gameweeks = normalize_gameweeks(bootstrap["events"])
    players = normalize_players(bootstrap["elements"], clubs, positions)
    fixture_table = normalize_fixtures(fixtures, clubs)
    return {
        "players": players,
        "clubs": clubs,
        "positions": positions,
        "gameweeks": gameweeks,
        "fixtures": fixture_table,
    }


def normalize_players(
    records: Sequence[Mapping[str, Any]],
    clubs: pd.DataFrame,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """Create a player table, retaining all API fields and adding readable values."""
    _require_record_fields(records, PLAYER_REQUIRED_FIELDS, "player")
    players = pd.DataFrame(records).rename(columns=PLAYER_RENAMES)

    club_names = clubs.set_index("club_id")["club_name"]
    club_short_names = clubs.set_index("club_id")["club_short_name"]
    position_names = positions.set_index("position_id")["position"]
    position_short_names = positions.set_index("position_id")["position_short_name"]

    players["club_name"] = players["club_id"].map(club_names)
    players["club_short_name"] = players["club_id"].map(club_short_names)
    players["position"] = players["position_id"].map(position_names)
    players["position_short_name"] = players["position_id"].map(
        position_short_names
    )
    _raise_for_unmapped(players, "club_name", "club_id", "club")
    _raise_for_unmapped(players, "position", "position_id", "position")

    first_names = players["first_name"].fillna("").astype(str).str.strip()
    surnames = players["surname"].fillna("").astype(str).str.strip()
    players["full_name"] = (first_names + " " + surnames).str.strip()
    players["display_name"] = players["display_name"].fillna(players["full_name"])

    raw_prices = pd.to_numeric(players["price"], errors="coerce")
    if raw_prices.isna().any():
        bad_ids = players.loc[raw_prices.isna(), "player_id"].tolist()
        raise NormalizationError(f"Non-numeric player prices for IDs: {bad_ids}")
    players["price"] = raw_prices / 10.0

    players["photo_url"] = players["player_code"].map(
        lambda value: PHOTO_URL_TEMPLATE.format(player_code=value)
    )
    if "ownership_percent" in players:
        players["ownership_percent"] = pd.to_numeric(
            players["ownership_percent"], errors="coerce"
        )

    id_columns = ["player_id", "player_code", "club_id", "position_id"]
    for column in id_columns:
        players[column] = pd.to_numeric(players[column], errors="raise").astype(
            "int64"
        )

    ordered = [column for column in PLAYER_FRONT_COLUMNS if column in players]
    remaining = [column for column in players.columns if column not in ordered]
    return players.loc[:, ordered + remaining].sort_values(
        "player_id", ignore_index=True
    )


def normalize_clubs(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Create a club table while retaining fields supplied by the API."""
    _require_record_fields(records, {"id", "name", "short_name"}, "club")
    clubs = pd.DataFrame(records).rename(
        columns={
            "id": "club_id",
            "name": "club_name",
            "short_name": "club_short_name",
        }
    )
    clubs["club_id"] = pd.to_numeric(clubs["club_id"], errors="raise").astype(
        "int64"
    )
    return clubs.sort_values("club_id", ignore_index=True)


def normalize_positions(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Create a position table with readable long and short names."""
    _require_record_fields(
        records, {"id", "singular_name", "singular_name_short"}, "position"
    )
    positions = pd.DataFrame(records).rename(
        columns={
            "id": "position_id",
            "singular_name": "position",
            "singular_name_short": "position_short_name",
        }
    )
    positions["position_id"] = pd.to_numeric(
        positions["position_id"], errors="raise"
    ).astype("int64")
    return positions.sort_values("position_id", ignore_index=True)


def normalize_gameweeks(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Create a gameweek table."""
    _require_record_fields(records, {"id", "name"}, "gameweek")
    gameweeks = pd.DataFrame(records).rename(
        columns={"id": "gameweek_id", "name": "gameweek_name"}
    )
    gameweeks["gameweek_id"] = pd.to_numeric(
        gameweeks["gameweek_id"], errors="raise"
    ).astype("int64")
    return gameweeks.sort_values("gameweek_id", ignore_index=True)


def normalize_fixtures(
    records: Sequence[Mapping[str, Any]],
    clubs: pd.DataFrame,
) -> pd.DataFrame:
    """Create a fixture table with readable home and away club names."""
    if not records:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "gameweek_id",
                "home_club_id",
                "home_club_name",
                "away_club_id",
                "away_club_name",
            ]
        )

    _require_record_fields(records, {"id", "team_h", "team_a"}, "fixture")
    fixture_table = pd.DataFrame(records).rename(
        columns={
            "id": "fixture_id",
            "event": "gameweek_id",
            "team_h": "home_club_id",
            "team_a": "away_club_id",
        }
    )
    club_names = clubs.set_index("club_id")["club_name"]
    fixture_table["home_club_name"] = fixture_table["home_club_id"].map(club_names)
    fixture_table["away_club_name"] = fixture_table["away_club_id"].map(club_names)
    _raise_for_unmapped(fixture_table, "home_club_name", "home_club_id", "home club")
    _raise_for_unmapped(fixture_table, "away_club_name", "away_club_id", "away club")

    for column in ("fixture_id", "home_club_id", "away_club_id"):
        fixture_table[column] = pd.to_numeric(
            fixture_table[column], errors="raise"
        ).astype("int64")
    if "gameweek_id" in fixture_table:
        fixture_table["gameweek_id"] = pd.to_numeric(
            fixture_table["gameweek_id"], errors="coerce"
        ).astype("Int64")

    front = [
        "fixture_id",
        "gameweek_id",
        "kickoff_time",
        "home_club_id",
        "home_club_name",
        "away_club_id",
        "away_club_name",
    ]
    ordered = [column for column in front if column in fixture_table]
    remaining = [column for column in fixture_table.columns if column not in ordered]
    return fixture_table.loc[:, ordered + remaining].sort_values(
        "fixture_id", ignore_index=True
    )


def _require_record_fields(
    records: Sequence[Mapping[str, Any]],
    required: set[str],
    record_name: str,
) -> None:
    for index, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            raise NormalizationError(
                f"{record_name.title()} record {index} is missing fields: "
                + ", ".join(sorted(missing))
            )


def _raise_for_unmapped(
    frame: pd.DataFrame,
    mapped_column: str,
    id_column: str,
    label: str,
) -> None:
    missing = frame[mapped_column].isna()
    if missing.any():
        ids = sorted(frame.loc[missing, id_column].unique().tolist())
        raise NormalizationError(f"Unknown {label} IDs: {ids}")

