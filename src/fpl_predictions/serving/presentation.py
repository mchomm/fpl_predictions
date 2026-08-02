"""Presentation helpers for the interactive FPL squad builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.schemas import SquadSelection

POSITION_ORDER = ("GKP", "DEF", "MID", "FWD")


def legal_formations(rules: SquadRules) -> tuple[str, ...]:
    """Return every formation allowed by the current API-derived rules."""
    positions = rules.position_by_short_name
    formations: list[str] = []
    for defenders in range(
        positions["DEF"].minimum_starters,
        positions["DEF"].maximum_starters + 1,
    ):
        for midfielders in range(
            positions["MID"].minimum_starters,
            positions["MID"].maximum_starters + 1,
        ):
            forwards = rules.starting_size - 1 - defenders - midfielders
            if (
                positions["FWD"].minimum_starters
                <= forwards
                <= positions["FWD"].maximum_starters
            ):
                formations.append(f"{defenders}-{midfielders}-{forwards}")
    return tuple(formations)


def formation_counts(formation: str, rules: SquadRules) -> dict[str, int]:
    """Parse and validate a formation into starting position counts."""
    try:
        defenders, midfielders, forwards = (
            int(value) for value in formation.split("-")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid formation: {formation!r}") from exc
    result = {
        "GKP": 1,
        "DEF": defenders,
        "MID": midfielders,
        "FWD": forwards,
    }
    if formation not in legal_formations(rules):
        raise ValueError(f"Formation {formation!r} is not legal")
    return result


def slot_positions(
    formation: str,
    rules: SquadRules,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ordered starter and substitute positions for a formation."""
    starters = formation_counts(formation, rules)
    starter_positions = tuple(
        position
        for position in POSITION_ORDER
        for _ in range(starters[position])
    )
    bench_positions = tuple(
        position
        for position in POSITION_ORDER
        for _ in range(
            rules.position_by_short_name[position].squad_count
            - starters[position]
        )
    )
    return starter_positions, bench_positions


def selection_defaults(
    selection: SquadSelection | None,
    formation: str,
    players: pd.DataFrame,
    rules: SquadRules,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Fit an existing squad into a new formation, using zero for blank slots."""
    starter_positions, bench_positions = slot_positions(formation, rules)
    if selection is None:
        return (
            tuple(0 for _ in starter_positions),
            tuple(0 for _ in bench_positions),
        )
    position_lookup = (
        players.drop_duplicates("player_id")
        .set_index("player_id")["position_short_name"]
        .astype(str)
        .to_dict()
    )
    preferred: dict[str, list[int]] = {position: [] for position in POSITION_ORDER}
    for player_id in selection.starting_xi + selection.bench:
        position = position_lookup.get(int(player_id))
        if position in preferred and int(player_id) not in preferred[position]:
            preferred[position].append(int(player_id))

    starters: list[int] = []
    used: set[int] = set()
    for position in starter_positions:
        candidate = next(
            (player for player in preferred[position] if player not in used),
            0,
        )
        starters.append(candidate)
        if candidate:
            used.add(candidate)
    bench: list[int] = []
    for position in bench_positions:
        candidate = next(
            (player for player in preferred[position] if player not in used),
            0,
        )
        bench.append(candidate)
        if candidate:
            used.add(candidate)
    return tuple(starters), tuple(bench)


def build_manual_selection(
    starters: Sequence[int],
    bench: Sequence[int],
    captain: int,
    vice_captain: int,
    *,
    bank: float | None,
    budget_limit: float | None,
    source: str = "manual",
) -> SquadSelection:
    """Build a strict selection from completed interactive pitch slots."""
    values = tuple(int(value) for value in tuple(starters) + tuple(bench))
    if any(value <= 0 for value in values):
        raise ValueError("Choose a player for every starting and substitute slot")
    if int(captain) <= 0 or int(vice_captain) <= 0:
        raise ValueError("Choose both a captain and vice-captain")
    return SquadSelection(
        player_ids=values,
        starting_xi=tuple(int(value) for value in starters),
        bench=tuple(int(value) for value in bench),
        captain=int(captain),
        vice_captain=int(vice_captain),
        source=source,  # type: ignore[arg-type]
        bank=bank,
        budget_limit=budget_limit,
    )


def player_visual_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return browser-safe visual fields with optional portrait support."""
    photo = row.get("photo_url")
    return {
        "player_id": int(row["player_id"]),
        "display_name": str(row["display_name"]),
        "club_name": str(row["club_name"]),
        "club_short_name": str(row.get("club_short_name") or ""),
        "position": str(row["position_short_name"]),
        "price": float(row["price"]),
        "photo_url": str(photo) if pd.notna(photo) and photo else None,
    }
