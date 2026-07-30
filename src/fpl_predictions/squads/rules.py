"""FPL squad rules derived from the current API response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SquadRulesError(ValueError):
    """Raised when the API cannot supply a complete squad rule set."""


@dataclass(frozen=True, slots=True)
class PositionRule:
    """Squad quota and starting limits for one API-defined position."""

    position_id: int
    name: str
    short_name: str
    squad_count: int
    minimum_starters: int
    maximum_starters: int


@dataclass(frozen=True, slots=True)
class SquadRules:
    """Current legal FPL squad constraints."""

    squad_size: int
    starting_size: int
    budget: float
    max_players_per_club: int
    positions: tuple[PositionRule, ...]

    @classmethod
    def from_bootstrap(cls, bootstrap: dict[str, Any]) -> "SquadRules":
        """Derive all supported constraints from bootstrap data."""
        settings = bootstrap.get("game_settings")
        position_rows = bootstrap.get("element_types")
        if not isinstance(settings, dict):
            raise SquadRulesError("bootstrap game_settings must be an object")
        if not isinstance(position_rows, list) or not position_rows:
            raise SquadRulesError(
                "bootstrap element_types must be a non-empty list"
            )
        required_settings = {
            "squad_squadsize",
            "squad_squadplay",
            "squad_team_limit",
            "squad_total_spend",
            "ui_currency_multiplier",
        }
        missing = required_settings.difference(settings)
        if missing:
            raise SquadRulesError(
                "game_settings is missing fields: "
                + ", ".join(sorted(missing))
            )
        multiplier = _positive_number(
            settings["ui_currency_multiplier"],
            "ui_currency_multiplier",
        )
        positions = []
        for index, row in enumerate(position_rows):
            if not isinstance(row, dict):
                raise SquadRulesError(
                    f"element_types row {index} must be an object"
                )
            required = {
                "id",
                "singular_name",
                "singular_name_short",
                "squad_select",
                "squad_min_play",
                "squad_max_play",
            }
            missing_position = required.difference(row)
            if missing_position:
                raise SquadRulesError(
                    f"element_types row {index} is missing fields: "
                    + ", ".join(sorted(missing_position))
                )
            positions.append(
                PositionRule(
                    position_id=_positive_int(row["id"], "position id"),
                    name=str(row["singular_name"]),
                    short_name=str(row["singular_name_short"]),
                    squad_count=_positive_int(
                        row["squad_select"], "squad_select"
                    ),
                    minimum_starters=_positive_int(
                        row["squad_min_play"], "squad_min_play"
                    ),
                    maximum_starters=_positive_int(
                        row["squad_max_play"], "squad_max_play"
                    ),
                )
            )
        rules = cls(
            squad_size=_positive_int(
                settings["squad_squadsize"], "squad_squadsize"
            ),
            starting_size=_positive_int(
                settings["squad_squadplay"], "squad_squadplay"
            ),
            budget=_positive_number(
                settings["squad_total_spend"], "squad_total_spend"
            )
            / multiplier,
            max_players_per_club=_positive_int(
                settings["squad_team_limit"], "squad_team_limit"
            ),
            positions=tuple(positions),
        )
        if sum(rule.squad_count for rule in rules.positions) != rules.squad_size:
            raise SquadRulesError(
                "Position squad quotas do not sum to the API squad size"
            )
        return rules

    @property
    def position_by_id(self) -> dict[int, PositionRule]:
        return {position.position_id: position for position in self.positions}

    @property
    def position_by_short_name(self) -> dict[str, PositionRule]:
        return {position.short_name: position for position in self.positions}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SquadRulesError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SquadRulesError(f"{name} must be a positive number")
    return float(value)

