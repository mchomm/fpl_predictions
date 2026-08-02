"""Side-effect-free rating and recommendation services for web consumers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl_predictions.serving.bundle import ServingBundle
from fpl_predictions.squads.optimizer import OptimizationResult, optimize_squad
from fpl_predictions.squads.projection import (
    player_projection_details,
    project_squad,
)
from fpl_predictions.squads.ratings import SquadRating, rate_squad
from fpl_predictions.squads.schemas import SquadSelection
from fpl_predictions.squads.validation import ValidatedSquad, validate_squad


def rate_selection(
    bundle: ServingBundle,
    selection: SquadSelection,
    horizon: int,
) -> tuple[ValidatedSquad, Any, SquadRating, list[dict[str, Any]]]:
    """Validate, project, and rate a selection from immutable serving data."""
    if horizon not in bundle.references:
        raise ValueError(f"Serving bundle has no horizon-{horizon} references")
    validated = validate_squad(selection, bundle.players, bundle.rules)
    projection = project_squad(
        validated, bundle.predictions, bundle.rules, horizon
    )
    rating = rate_squad(
        projection,
        bundle.references[horizon],
        (
            "ownership-conditioned, budget-matched legal reference population "
            f"(horizon={horizon})"
        ),
    )
    details = player_projection_details(validated, bundle.predictions, horizon)
    return validated, projection, rating, details


def optimize_selection(
    bundle: ServingBundle,
    horizon: int,
    *,
    current_squad: SquadSelection | None = None,
    max_transfers: int | None = None,
    free_transfers: int = 1,
    hit_cost: float = 4.0,
    bench_weight: float = 0.1,
) -> OptimizationResult:
    """Build a best squad or recommend net-positive bounded transfers."""
    return optimize_squad(
        bundle.players,
        bundle.predictions,
        bundle.rules,
        horizon,
        current_squad=current_squad,
        max_transfers=max_transfers,
        free_transfers=free_transfers,
        hit_cost=hit_cost,
        bench_weight=bench_weight,
    )


def player_label_lookup(players: pd.DataFrame) -> tuple[dict[int, str], dict[str, int]]:
    """Return stable ID/label mappings suitable for select widgets."""
    labels: dict[int, str] = {}
    reverse: dict[str, int] = {}
    for row in players.sort_values(
        ["position_id", "price", "display_name"],
        ascending=[True, False, True],
    ).itertuples():
        label = (
            f"{row.display_name} — {row.club_name} · "
            f"{row.position_short_name} · £{float(row.price):.1f}m "
            f"[ID {int(row.player_id)}]"
        )
        labels[int(row.player_id)] = label
        reverse[label] = int(row.player_id)
    return labels, reverse
