"""Exact mixed-integer optimization for FPL squads and transfers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from fpl_predictions.squads.projection import (
    SquadProjection,
    add_availability_adjusted_predictions,
    project_squad,
)
from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.schemas import SquadSelection
from fpl_predictions.squads.validation import validate_squad


class SquadOptimizationError(RuntimeError):
    """Raised when no legal optimum can be produced."""


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Auditable optimized selection and optional transfer comparison."""

    selection: SquadSelection
    projection: SquadProjection
    objective_points: float
    budget: float
    total_cost: float
    bank_remaining: float
    transfers_in: tuple[int, ...]
    transfers_out: tuple[int, ...]
    baseline_projection: SquadProjection | None
    projected_points_gain: float | None
    free_transfers: int
    paid_transfers: int
    transfer_hit_points: float
    net_projected_points_gain: float | None
    budget_warnings: tuple[str, ...]
    solver_message: str
    mip_gap: float | None

    def as_dict(
        self,
        players: pd.DataFrame,
        predictions: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        indexed = players.drop_duplicates("player_id").set_index("player_id")
        prediction_indexed = (
            predictions.drop_duplicates("player_id").set_index("player_id")
            if predictions is not None
            else None
        )

        def player_record(player_id: int) -> dict[str, Any]:
            row = indexed.loc[player_id]
            record = {
                "player_id": int(player_id),
                "display_name": str(row.get("display_name", player_id)),
                "club_name": str(row.get("club_name", "")),
                "position": str(row.get("position_short_name", "")),
                "price": float(row["price"]),
            }
            prediction_column = f"predicted_points_{self.projection.horizon}"
            if (
                prediction_indexed is not None
                and player_id in prediction_indexed.index
                and prediction_column in prediction_indexed
            ):
                record["raw_model_points"] = float(
                    prediction_indexed.loc[player_id, prediction_column]
                )
            return record

        incoming_records = [player_record(value) for value in self.transfers_in]
        outgoing_records = [player_record(value) for value in self.transfers_out]

        def sum_field(records: list[dict[str, Any]], field: str) -> float:
            return float(sum(float(record.get(field, 0.0)) for record in records))

        return {
            "selection": self.selection.as_dict(),
            "selected_players": [
                player_record(value) for value in self.selection.player_ids
            ],
            "starting_xi": [
                player_record(value) for value in self.selection.starting_xi
            ],
            "bench": [player_record(value) for value in self.selection.bench],
            "captain": player_record(self.selection.captain),
            "vice_captain": player_record(self.selection.vice_captain),
            "projection": self.projection.as_dict(),
            "objective_points": self.objective_points,
            "budget": self.budget,
            "total_cost": self.total_cost,
            "bank_remaining": self.bank_remaining,
            "transfers_in": incoming_records,
            "transfers_out": outgoing_records,
            "transfer_summary": {
                "count": len(incoming_records),
                "incoming_current_price": sum_field(incoming_records, "price"),
                "outgoing_current_price": sum_field(outgoing_records, "price"),
                "current_price_difference": (
                    sum_field(incoming_records, "price")
                    - sum_field(outgoing_records, "price")
                ),
                "incoming_raw_model_points": sum_field(
                    incoming_records, "raw_model_points"
                ),
                "outgoing_raw_model_points": sum_field(
                    outgoing_records, "raw_model_points"
                ),
                "squad_projected_points_gain": self.projected_points_gain,
                "free_transfers": self.free_transfers,
                "paid_transfers": self.paid_transfers,
                "transfer_hit_points": self.transfer_hit_points,
                "net_projected_points_gain": self.net_projected_points_gain,
                "note": (
                    "Squad gain can differ from the player-point difference "
                    "because the optimizer also changes formation, bench order, "
                    "captaincy, and the starting XI."
                ),
            },
            "baseline_projection": (
                self.baseline_projection.as_dict()
                if self.baseline_projection is not None
                else None
            ),
            "projected_points_gain": self.projected_points_gain,
            "net_projected_points_gain": self.net_projected_points_gain,
            "budget_warnings": list(self.budget_warnings),
            "solver": {
                "method": "scipy.optimize.milp (HiGHS)",
                "message": self.solver_message,
                "mip_gap": self.mip_gap,
            },
        }


def optimize_squad(
    players: pd.DataFrame,
    predictions: pd.DataFrame,
    rules: SquadRules,
    horizon: int = 3,
    *,
    current_squad: SquadSelection | None = None,
    max_transfers: int | None = None,
    budget: float | None = None,
    bench_weight: float = 0.1,
    locked_player_ids: Iterable[int] = (),
    excluded_player_ids: Iterable[int] = (),
    time_limit_seconds: float = 60.0,
    free_transfers: int = 1,
    hit_cost: float = 4.0,
) -> OptimizationResult:
    """Maximize projected XI, captain and weighted bench under exact rules."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= bench_weight <= 1:
        raise ValueError("bench_weight must be between 0 and 1")
    if max_transfers is not None and max_transfers < 0:
        raise ValueError("max_transfers must be non-negative")
    if current_squad is None and max_transfers is not None:
        raise ValueError("max_transfers requires a current_squad")
    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    if current_squad is None and free_transfers != 1:
        raise ValueError("free_transfers only applies to an existing squad")
    if not 0 <= free_transfers <= 5:
        raise ValueError("free_transfers must be between 0 and 5")
    if hit_cost < 0:
        raise ValueError("hit_cost must be non-negative")

    current_validated = (
        validate_squad(current_squad, players, rules)
        if current_squad is not None
        else None
    )
    selling_prices, budget_warnings = _current_selling_prices(
        players, current_squad
    )
    resolved_budget = _resolve_budget(
        rules,
        current_squad,
        current_validated.total_cost if current_validated is not None else None,
        budget,
        selling_prices,
    )
    current_ids = set(current_squad.player_ids) if current_squad is not None else set()
    candidates = _candidate_table(players, predictions, horizon, current_ids)
    locked = set(int(value) for value in locked_player_ids)
    excluded = set(int(value) for value in excluded_player_ids)
    if locked & excluded:
        raise ValueError("A player cannot be both locked and excluded")
    missing_locked = locked.difference(candidates["player_id"])
    if missing_locked:
        raise ValueError(f"Locked player IDs lack prediction coverage: {sorted(missing_locked)}")

    solution, solver_result = _solve_milp(
        candidates,
        rules,
        resolved_budget,
        bench_weight,
        current_ids,
        max_transfers,
        locked,
        excluded,
        time_limit_seconds,
        selling_prices,
        free_transfers,
        hit_cost,
    )
    selected = candidates.loc[solution["squad"]].copy()
    starters = candidates.loc[solution["starter"]].copy()
    captain = int(candidates.loc[solution["captain"], "player_id"].iloc[0])
    vice = int(
        starters.loc[starters["player_id"] != captain]
        .sort_values(
            ["availability_adjusted_points", "player_id"],
            ascending=[False, True],
        )["player_id"]
        .iloc[0]
    )
    player_ids = tuple(
        selected.sort_values(["position_id", "player_id"])["player_id"].astype(int)
    )
    starting_xi = tuple(
        starters.sort_values(["position_id", "player_id"])["player_id"].astype(int)
    )
    starter_set = set(starting_xi)
    bench_rows = selected.loc[~selected["player_id"].isin(starter_set)].copy()
    goalkeeper_ids = {
        item.position_id
        for item in rules.positions
        if item.short_name.upper() == "GKP"
    }
    bench_rows["_goalkeeper_first"] = ~bench_rows["position_id"].isin(
        goalkeeper_ids
    )
    bench = tuple(
        bench_rows.sort_values(
            ["_goalkeeper_first", "availability_adjusted_points", "player_id"],
            ascending=[True, False, True],
        )["player_id"].astype(int)
    )
    selected_budget_cost = float(
        sum(
            selling_prices.get(int(row.player_id), float(row.price))
            for row in selected.itertuples()
        )
    )
    bank_remaining = max(resolved_budget - selected_budget_cost, 0.0)
    selection = SquadSelection(
        player_ids=player_ids,
        starting_xi=starting_xi,
        bench=bench,
        captain=captain,
        vice_captain=vice,
        source="generated",
        bank=bank_remaining,
        budget_limit=resolved_budget,
    )
    validated = validate_squad(selection, players, rules)
    projection = project_squad(validated, predictions, rules, horizon)
    baseline = (
        project_squad(current_validated, predictions, rules, horizon)
        if current_validated is not None
        else None
    )
    transfers_in = (
        tuple(sorted(set(player_ids).difference(current_ids)))
        if current_squad is not None
        else ()
    )
    transfers_out = (
        tuple(sorted(current_ids.difference(player_ids)))
        if current_squad is not None
        else ()
    )
    paid_transfers = max(len(transfers_in) - free_transfers, 0)
    transfer_hit_points = float(paid_transfers * hit_cost)
    projected_gain = (
        projection.overall_points - baseline.overall_points
        if baseline is not None
        else None
    )
    return OptimizationResult(
        selection=selection,
        projection=projection,
        objective_points=float(-solver_result.fun),
        budget=resolved_budget,
        total_cost=validated.total_cost,
        bank_remaining=bank_remaining,
        transfers_in=transfers_in,
        transfers_out=transfers_out,
        baseline_projection=baseline,
        projected_points_gain=projected_gain,
        free_transfers=free_transfers if current_squad is not None else 0,
        paid_transfers=paid_transfers,
        transfer_hit_points=transfer_hit_points,
        net_projected_points_gain=(
            projected_gain - transfer_hit_points
            if projected_gain is not None
            else None
        ),
        budget_warnings=budget_warnings,
        solver_message=str(solver_result.message),
        mip_gap=(
            float(solver_result.mip_gap)
            if getattr(solver_result, "mip_gap", None) is not None
            else None
        ),
    )


def _candidate_table(
    players: pd.DataFrame,
    predictions: pd.DataFrame,
    horizon: int,
    current_ids: set[int],
) -> pd.DataFrame:
    required_players = {"player_id", "club_id", "position_id", "price"}
    missing = required_players.difference(players.columns)
    if missing:
        raise ValueError("Player table is missing: " + ", ".join(sorted(missing)))
    merged = players.merge(
        predictions,
        on="player_id",
        how="inner",
        suffixes=("", "_prediction"),
        validate="one_to_one",
    )
    if "can_select" in merged:
        selectable = merged["can_select"].fillna(True).astype(bool)
        merged = merged.loc[selectable | merged["player_id"].isin(current_ids)]
    adjusted, _ = add_availability_adjusted_predictions(merged, horizon)
    adjusted["price"] = pd.to_numeric(adjusted["price"], errors="coerce")
    adjusted["availability_adjusted_points"] = pd.to_numeric(
        adjusted["availability_adjusted_points"], errors="coerce"
    )
    valid = adjusted["price"].notna() & adjusted[
        "availability_adjusted_points"
    ].notna()
    return adjusted.loc[valid].sort_values("player_id").reset_index(drop=True)


def _resolve_budget(
    rules: SquadRules,
    current: SquadSelection | None,
    current_cost: float | None,
    explicit: float | None,
    selling_prices: dict[int, float],
) -> float:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("budget must be positive")
        return float(explicit)
    if current is None:
        return rules.budget
    if current.budget_limit is not None:
        return current.budget_limit
    assert current_cost is not None
    liquidation_value = float(sum(selling_prices.values()))
    if current.bank is not None:
        return liquidation_value + current.bank
    if current.source in {"manual", "screenshot"}:
        inferred_bank = max(rules.budget - current_cost, 0.0)
        return liquidation_value + inferred_bank
    return liquidation_value


def _current_selling_prices(
    players: pd.DataFrame,
    current: SquadSelection | None,
) -> tuple[dict[int, float], tuple[str, ...]]:
    """Resolve auditable sell values, preferring API values over reconstruction."""
    if current is None:
        return {}, ()
    indexed = players.drop_duplicates("player_id").set_index("player_id")
    direct = dict(current.selling_prices or {})
    purchases = dict(current.purchase_prices or {})
    resolved: dict[int, float] = {}
    approximated: list[int] = []
    reconstructed: list[int] = []
    for player_id in current.player_ids:
        current_price = float(indexed.loc[player_id, "price"])
        if player_id in direct:
            resolved[player_id] = float(direct[player_id])
        elif player_id in purchases:
            purchase_ticks = round(float(purchases[player_id]) * 10)
            current_ticks = round(current_price * 10)
            sell_ticks = (
                current_ticks
                if current_ticks <= purchase_ticks
                else purchase_ticks + (current_ticks - purchase_ticks) // 2
            )
            resolved[player_id] = sell_ticks / 10.0
            reconstructed.append(player_id)
        else:
            resolved[player_id] = current_price
            approximated.append(player_id)
    warnings = []
    if reconstructed:
        warnings.append(
            "Selling prices reconstructed from purchase and current prices for "
            f"player IDs {reconstructed}."
        )
    if approximated:
        warnings.append(
            "Current prices used as approximate selling prices for player IDs "
            f"{approximated}; supply selling_prices for exact affordability."
        )
    return resolved, tuple(warnings)


def _solve_milp(
    candidates: pd.DataFrame,
    rules: SquadRules,
    budget: float,
    bench_weight: float,
    current_ids: set[int],
    max_transfers: int | None,
    locked: set[int],
    excluded: set[int],
    time_limit_seconds: float,
    selling_prices: dict[int, float],
    free_transfers: int,
    hit_cost: float,
) -> tuple[dict[str, np.ndarray], Any]:
    count = len(candidates)
    if count == 0:
        raise SquadOptimizationError("No prediction-covered candidates are available")
    squad_slice = slice(0, count)
    starter_slice = slice(count, 2 * count)
    captain_slice = slice(2 * count, 3 * count)
    paid_transfer_index = 3 * count if current_ids else None
    variable_count = 3 * count + (1 if current_ids else 0)
    points = candidates["availability_adjusted_points"].to_numpy(dtype=float)
    objective = np.zeros(variable_count, dtype=float)
    objective[squad_slice] = -bench_weight * points
    objective[starter_slice] = -(1.0 - bench_weight) * points
    objective[captain_slice] = -points
    if paid_transfer_index is not None:
        objective[paid_transfer_index] = hit_cost
    # Stable tie-break far below the stored projection precision.
    objective[squad_slice] += candidates["player_id"].to_numpy(dtype=float) * 1e-10

    rows: list[tuple[dict[int, float], float, float]] = []

    def add(coefficients: dict[int, float], lower: float, upper: float) -> None:
        rows.append((coefficients, lower, upper))

    add({index: 1.0 for index in range(count)}, rules.squad_size, rules.squad_size)
    add(
        {count + index: 1.0 for index in range(count)},
        rules.starting_size,
        rules.starting_size,
    )
    add({2 * count + index: 1.0 for index in range(count)}, 1.0, 1.0)
    for index in range(count):
        add({count + index: 1.0, index: -1.0}, -np.inf, 0.0)
        add({2 * count + index: 1.0, count + index: -1.0}, -np.inf, 0.0)

    for position in rules.positions:
        indexes = candidates.index[candidates["position_id"] == position.position_id]
        add(
            {int(index): 1.0 for index in indexes},
            position.squad_count,
            position.squad_count,
        )
        add(
            {count + int(index): 1.0 for index in indexes},
            position.minimum_starters,
            position.maximum_starters,
        )
    for _, group in candidates.groupby("club_id"):
        add(
            {int(index): 1.0 for index in group.index},
            -np.inf,
            rules.max_players_per_club,
        )
    budget_coefficients = {
        int(index): selling_prices.get(int(row.player_id), float(row.price))
        for index, row in enumerate(candidates.itertuples())
    }
    add(
        budget_coefficients,
        -np.inf,
        budget,
    )
    if current_ids and max_transfers is not None:
        retained = candidates.index[candidates["player_id"].isin(current_ids)]
        add(
            {int(index): 1.0 for index in retained},
            rules.squad_size - max_transfers,
            np.inf,
        )
    if current_ids and paid_transfer_index is not None:
        retained = candidates.index[candidates["player_id"].isin(current_ids)]
        coefficients = {int(index): 1.0 for index in retained}
        coefficients[paid_transfer_index] = 1.0
        add(
            coefficients,
            rules.squad_size - free_transfers,
            np.inf,
        )

    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.ones(variable_count, dtype=float)
    if paid_transfer_index is not None:
        upper_bounds[paid_transfer_index] = rules.squad_size
    id_to_index = dict(zip(candidates["player_id"], candidates.index, strict=True))
    for player_id in locked:
        lower_bounds[int(id_to_index[player_id])] = 1.0
    for player_id in excluded:
        index = id_to_index.get(player_id)
        if index is not None:
            upper_bounds[int(index)] = 0.0

    matrix = lil_matrix((len(rows), variable_count), dtype=float)
    lower = np.empty(len(rows), dtype=float)
    upper = np.empty(len(rows), dtype=float)
    for row_index, (coefficients, row_lower, row_upper) in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
        lower[row_index] = row_lower
        upper[row_index] = row_upper
    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=int),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={
            "time_limit": time_limit_seconds,
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    if not result.success or result.x is None:
        raise SquadOptimizationError(
            "No legal optimized squad was found: " + str(result.message)
        )
    binary = result.x >= 0.5
    return {
        "squad": binary[squad_slice],
        "starter": binary[starter_slice],
        "captain": binary[captain_slice],
    }, result
