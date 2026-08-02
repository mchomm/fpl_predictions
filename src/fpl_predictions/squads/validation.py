"""Validation of complete FPL squads against API-derived rules."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.schemas import SquadSelection


class SquadValidationError(ValueError):
    """Raised with all discovered legality errors for a squad."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("Invalid FPL squad: " + "; ".join(errors))


@dataclass(frozen=True, slots=True)
class ValidatedSquad:
    """A legal squad enriched with current cost and formation."""

    selection: SquadSelection
    formation: str
    total_cost: float
    budget_limit: float | None
    player_rows: pd.DataFrame


def validate_squad(
    squad: SquadSelection,
    players: pd.DataFrame,
    rules: SquadRules,
) -> ValidatedSquad:
    """Validate membership, positions, formation, clubs, captaincy, and budget."""
    required_columns = {
        "player_id",
        "club_id",
        "position_id",
        "position_short_name",
        "price",
    }
    missing_columns = required_columns.difference(players.columns)
    if missing_columns:
        raise ValueError(
            "Player table is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    errors: list[str] = []
    _check_length_and_uniqueness(
        squad.player_ids, rules.squad_size, "squad", errors
    )
    _check_length_and_uniqueness(
        squad.starting_xi, rules.starting_size, "starting XI", errors
    )
    bench_size = rules.squad_size - rules.starting_size
    _check_length_and_uniqueness(squad.bench, bench_size, "bench", errors)

    squad_set = set(squad.player_ids)
    xi_set = set(squad.starting_xi)
    bench_set = set(squad.bench)
    if xi_set & bench_set:
        errors.append("starting XI and bench must not overlap")
    if xi_set | bench_set != squad_set:
        errors.append(
            "starting XI and bench must contain exactly the 15 squad players"
        )
    if squad.captain == squad.vice_captain:
        errors.append("captain and vice-captain must be different players")
    if squad.captain not in xi_set:
        errors.append("captain must be in the starting XI")
    if squad.vice_captain not in xi_set:
        errors.append("vice-captain must be in the starting XI")
    if squad.active_chip is not None:
        errors.append(
            "active_chip scoring is not supported yet; rate the normal squad "
            "or omit active_chip"
        )

    indexed = players.drop_duplicates("player_id").set_index("player_id")
    stale_ids = sorted(squad_set.difference(indexed.index))
    if stale_ids:
        errors.append(f"unknown or stale player IDs: {stale_ids}")
    if errors:
        raise SquadValidationError(errors)

    selected = indexed.loc[list(squad.player_ids)].reset_index()
    position_counts = selected["position_id"].value_counts().to_dict()
    for position in rules.positions:
        actual = int(position_counts.get(position.position_id, 0))
        if actual != position.squad_count:
            errors.append(
                f"{position.short_name} squad count must be "
                f"{position.squad_count}, got {actual}"
            )

    club_counts = selected["club_id"].value_counts()
    excessive_clubs = club_counts[
        club_counts > rules.max_players_per_club
    ]
    if not excessive_clubs.empty:
        details = {
            int(club): int(count)
            for club, count in excessive_clubs.items()
        }
        errors.append(
            "club player limit exceeded "
            f"(maximum {rules.max_players_per_club}): {details}"
        )

    xi_rows = indexed.loc[list(squad.starting_xi)]
    xi_counts = xi_rows["position_id"].value_counts().to_dict()
    for position in rules.positions:
        actual = int(xi_counts.get(position.position_id, 0))
        if not position.minimum_starters <= actual <= position.maximum_starters:
            errors.append(
                f"{position.short_name} starters must be between "
                f"{position.minimum_starters} and "
                f"{position.maximum_starters}, got {actual}"
            )

    bench_rows = indexed.loc[list(squad.bench)]
    goalkeeper_ids = {
        rule.position_id
        for rule in rules.positions
        if rule.minimum_starters == rule.maximum_starters == 1
    }
    bench_goalkeepers = int(
        bench_rows["position_id"].isin(goalkeeper_ids).sum()
    )
    if bench_goalkeepers != 1:
        errors.append(
            f"bench must contain exactly one goalkeeper, got {bench_goalkeepers}"
        )

    prices = pd.to_numeric(selected["price"], errors="coerce")
    if prices.isna().any():
        errors.append("all squad players must have a current numeric price")
        total_cost = float("nan")
    else:
        total_cost = float(prices.sum())
    budget_limit = squad.budget_limit
    if budget_limit is None and squad.source in {"manual", "generated"}:
        budget_limit = rules.budget
    if (
        budget_limit is not None
        and pd.notna(total_cost)
        and total_cost > budget_limit + 1e-9
    ):
        errors.append(
            f"squad cost {total_cost:.1f} exceeds budget {budget_limit:.1f}"
        )
    if errors:
        raise SquadValidationError(errors)

    formation_counts = {
        rule.short_name: int(xi_counts.get(rule.position_id, 0))
        for rule in rules.positions
    }
    outfield = [
        formation_counts[rule.short_name]
        for rule in rules.positions
        if rule.position_id not in goalkeeper_ids
    ]
    formation = "-".join(str(value) for value in outfield)
    return ValidatedSquad(
        selection=squad,
        formation=formation,
        total_cost=total_cost,
        budget_limit=budget_limit,
        player_rows=selected,
    )


def _check_length_and_uniqueness(
    values: tuple[int, ...],
    expected: int,
    label: str,
    errors: list[str],
) -> None:
    if len(values) != expected:
        errors.append(f"{label} must contain {expected} players, got {len(values)}")
    if len(set(values)) != len(values):
        errors.append(f"{label} contains duplicate players")
