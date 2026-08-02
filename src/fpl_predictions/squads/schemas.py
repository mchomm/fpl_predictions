"""Explicit input schemas for reconstructed FPL squads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

SquadSource = Literal["manual", "manager_api", "screenshot", "generated"]
VALID_SOURCES = {"manual", "manager_api", "screenshot", "generated"}


class SquadSchemaError(ValueError):
    """Raised when squad input cannot be parsed into an explicit schema."""


@dataclass(frozen=True, slots=True)
class SquadSelection:
    """A complete squad selection before current-rule validation."""

    player_ids: tuple[int, ...]
    starting_xi: tuple[int, ...]
    bench: tuple[int, ...]
    captain: int
    vice_captain: int
    source: SquadSource = "manual"
    bank: float | None = None
    budget_limit: float | None = None
    active_chip: str | None = None
    purchase_prices: Mapping[int, float] | None = None
    selling_prices: Mapping[int, float] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SquadSelection":
        """Parse JSON-compatible squad input with strict integer IDs."""
        required = {
            "player_ids",
            "starting_xi",
            "bench",
            "captain",
            "vice_captain",
        }
        missing = required.difference(value)
        if missing:
            raise SquadSchemaError(
                "Squad input is missing fields: "
                + ", ".join(sorted(missing))
            )
        source = value.get("source", "manual")
        if source not in VALID_SOURCES:
            raise SquadSchemaError(
                f"source must be one of: {', '.join(sorted(VALID_SOURCES))}"
            )
        return cls(
            player_ids=_id_tuple(value["player_ids"], "player_ids"),
            starting_xi=_id_tuple(value["starting_xi"], "starting_xi"),
            bench=_id_tuple(value["bench"], "bench"),
            captain=_player_id(value["captain"], "captain"),
            vice_captain=_player_id(
                value["vice_captain"], "vice_captain"
            ),
            source=source,
            bank=_optional_nonnegative_number(value.get("bank"), "bank"),
            budget_limit=_optional_positive_number(
                value.get("budget_limit"), "budget_limit"
            ),
            active_chip=(
                str(value["active_chip"])
                if value.get("active_chip") is not None
                else None
            ),
            purchase_prices=_optional_price_mapping(
                value.get("purchase_prices"), "purchase_prices"
            ),
            selling_prices=_optional_price_mapping(
                value.get("selling_prices"), "selling_prices"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "player_ids": list(self.player_ids),
            "starting_xi": list(self.starting_xi),
            "bench": list(self.bench),
            "captain": self.captain,
            "vice_captain": self.vice_captain,
            "source": self.source,
            "bank": self.bank,
            "budget_limit": self.budget_limit,
            "active_chip": self.active_chip,
            "purchase_prices": _json_price_mapping(self.purchase_prices),
            "selling_prices": _json_price_mapping(self.selling_prices),
        }


def _id_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise SquadSchemaError(f"{name} must be a list of player IDs")
    return tuple(_player_id(item, name) for item in value)


def _player_id(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SquadSchemaError(f"{name} must contain positive integer IDs")
    return value


def _optional_nonnegative_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise SquadSchemaError(f"{name} must be a non-negative number")
    return float(value)


def _optional_positive_number(value: Any, name: str) -> float | None:
    result = _optional_nonnegative_number(value, name)
    if result is not None and result <= 0:
        raise SquadSchemaError(f"{name} must be greater than zero")
    return result


def _optional_price_mapping(
    value: Any,
    name: str,
) -> dict[int, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SquadSchemaError(f"{name} must be an object keyed by player ID")
    result: dict[int, float] = {}
    for raw_player_id, raw_price in value.items():
        try:
            player_id = int(raw_player_id)
        except (TypeError, ValueError) as exc:
            raise SquadSchemaError(
                f"{name} keys must be positive player IDs"
            ) from exc
        if player_id <= 0 or str(player_id) != str(raw_player_id):
            raise SquadSchemaError(f"{name} keys must be positive player IDs")
        price = _optional_positive_number(raw_price, f"{name}[{player_id}]")
        assert price is not None
        result[player_id] = price
    return result


def _json_price_mapping(
    value: Mapping[int, float] | None,
) -> dict[str, float] | None:
    if value is None:
        return None
    return {str(player_id): float(price) for player_id, price in value.items()}
