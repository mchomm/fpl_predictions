"""Deterministic generation of legal comparison squads."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json

import numpy as np
import pandas as pd

from fpl_predictions.squads.projection import (
    add_availability_adjusted_predictions,
)
from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.schemas import SquadSelection
from fpl_predictions.squads.simulation import simulate_selection

REFERENCE_STRATEGIES = {
    "broad_legal",
    "human_like",
    "price_aware",
    "ownership_weighted",
}

PRICE_SCALE = 10
OWNERSHIP_RESOLUTION_PERCENT = 0.1


@dataclass(frozen=True, slots=True)
class _PositionSubsetSampler:
    """Exact weighted-subset sampler for one position and each total cost."""

    players: pd.DataFrame
    quota: int
    costs: np.ndarray
    weights: np.ndarray
    prefix_weights: np.ndarray

    @property
    def cost_distribution(self) -> np.ndarray:
        return self.prefix_weights[-1, self.quota]


def generate_reference_population(
    players: pd.DataFrame,
    predictions: pd.DataFrame,
    rules: SquadRules,
    horizon: int,
    size: int = 1000,
    strategy: str = "human_like",
    seed: int = 2026,
    target_cost: float | None = None,
    budget_band: float = 1.0,
    availability_simulations: int = 0,
) -> pd.DataFrame:
    """Generate legal squads and score their strongest predicted starting XI."""
    if size <= 0:
        raise ValueError("reference size must be positive")
    if strategy not in REFERENCE_STRATEGIES:
        raise ValueError(
            "strategy must be one of: " + ", ".join(sorted(REFERENCE_STRATEGIES))
        )
    if budget_band < 0:
        raise ValueError("budget_band must be non-negative")
    if target_cost is not None and target_cost <= 0:
        raise ValueError("target_cost must be positive")
    if availability_simulations < 0:
        raise ValueError("availability_simulations must be non-negative")
    merged = players.merge(
        predictions,
        on="player_id",
        how="inner",
        suffixes=("", "_prediction"),
        validate="one_to_one",
    )
    adjusted, _ = add_availability_adjusted_predictions(merged, horizon)
    rng = np.random.default_rng(seed)
    human_samplers: tuple[_PositionSubsetSampler, ...] | None = None
    human_cost_weights: np.ndarray | None = None
    minimum_cost = 0.0
    maximum_cost = rules.budget
    if strategy == "human_like":
        maximum_cost = min(
            rules.budget,
            rules.budget if target_cost is None else target_cost,
        )
        minimum_cost = max(0.0, maximum_cost - budget_band)
        human_samplers, human_cost_weights = _build_human_like_samplers(
            adjusted,
            rules,
            minimum_cost,
            maximum_cost,
        )
    records: list[dict[str, object]] = []
    attempts = 0
    maximum_attempts = max(5000, size * 500)
    while len(records) < size and attempts < maximum_attempts:
        attempts += 1
        if strategy == "human_like":
            assert human_samplers is not None
            assert human_cost_weights is not None
            player_ids = _draw_human_like_squad(
                human_samplers,
                human_cost_weights,
                rules,
                rng,
            )
        else:
            player_ids = _draw_squad(adjusted, rules, strategy, rng)
        if player_ids is None:
            continue
        selection = _best_selection(
            adjusted[adjusted["player_id"].isin(player_ids)],
            rules,
        )
        selected = adjusted[adjusted["player_id"].isin(player_ids)]
        if availability_simulations:
            simulation = simulate_selection(
                selected,
                selection,
                rules,
                horizon,
                simulations=availability_simulations,
                seed=seed,
            )
            record = simulation.projection.as_dict()
            record.pop("warnings", None)
            record.update(
                {
                    "simulation_count": availability_simulations,
                    "expected_autosub_points": simulation.expected_autosub_points,
                    "expected_vice_captain_points": (
                        simulation.expected_vice_captain_points
                    ),
                    "autosub_probability": simulation.autosub_probability,
                    "vice_captain_takeover_probability": (
                        simulation.vice_captain_takeover_probability
                    ),
                }
            )
        else:
            record = _reference_projection(selected, selection, rules, horizon)
        record.update(
            {
                "reference_index": len(records),
                "strategy": strategy,
                "seed": seed,
                "total_cost": float(selected["price"].sum()),
                "target_cost": maximum_cost if strategy == "human_like" else None,
                "minimum_reference_cost": (
                    minimum_cost if strategy == "human_like" else None
                ),
                "player_ids": json.dumps(list(selection.player_ids)),
                "starting_xi": json.dumps(list(selection.starting_xi)),
                "bench": json.dumps(list(selection.bench)),
                "captain": selection.captain,
                "vice_captain": selection.vice_captain,
            }
        )
        records.append(record)
    if len(records) != size:
        raise RuntimeError(
            f"Generated {len(records)} of {size} legal reference squads after "
            f"{attempts} attempts; check player coverage and prices"
        )
    return pd.DataFrame.from_records(records)


def _build_human_like_samplers(
    players: pd.DataFrame,
    rules: SquadRules,
    minimum_cost: float,
    maximum_cost: float,
) -> tuple[tuple[_PositionSubsetSampler, ...], np.ndarray]:
    """Build an ownership-derived distribution conditioned on comparable cost."""
    if "ownership_percent" not in players:
        raise ValueError(
            "human_like references require current ownership_percent data"
        )
    maximum_ticks = _price_ticks(maximum_cost)
    minimum_ticks = _price_ticks(minimum_cost)
    samplers = tuple(
        _build_position_sampler(
            players[players["position_id"] == position.position_id],
            position.squad_count,
            maximum_ticks,
        )
        for position in rules.positions
    )
    combined = np.zeros((len(samplers) + 1, maximum_ticks + 1), dtype=float)
    combined[0, 0] = 1.0
    for index, sampler in enumerate(samplers, start=1):
        convolution = np.convolve(
            combined[index - 1], sampler.cost_distribution
        )
        combined[index] = convolution[: maximum_ticks + 1]
    eligible = combined[-1].copy()
    eligible[:minimum_ticks] = 0.0
    if not np.isfinite(eligible).all() or eligible.sum() <= 0:
        raise ValueError(
            f"No ownership-weighted squads can be built between "
            f"{minimum_cost:.1f} and {maximum_cost:.1f}"
        )
    combined[-1] = eligible
    return samplers, combined


def _build_position_sampler(
    players: pd.DataFrame,
    quota: int,
    maximum_ticks: int,
) -> _PositionSubsetSampler:
    """Calculate exact subset weights by count and price using dynamic programming."""
    if len(players) < quota:
        raise ValueError("Not enough prediction-covered players for a position")
    frame = players.reset_index(drop=True)
    costs = np.array([_price_ticks(value) for value in frame["price"]], dtype=int)
    ownership = pd.to_numeric(
        frame["ownership_percent"], errors="coerce"
    ).fillna(0.0)
    weights = np.maximum(
        ownership.to_numpy(dtype=float), OWNERSHIP_RESOLUTION_PERCENT
    )
    weights /= weights.max()
    prefix = np.zeros(
        (len(frame) + 1, quota + 1, maximum_ticks + 1),
        dtype=float,
    )
    prefix[0, 0, 0] = 1.0
    for item_index, (cost, weight) in enumerate(
        zip(costs, weights, strict=True), start=1
    ):
        prefix[item_index] = prefix[item_index - 1]
        if cost > maximum_ticks:
            continue
        for count in range(1, quota + 1):
            prefix[item_index, count, cost:] += (
                weight * prefix[item_index - 1, count - 1, :-cost]
            )
    return _PositionSubsetSampler(
        players=frame,
        quota=quota,
        costs=costs,
        weights=weights,
        prefix_weights=prefix,
    )


def _draw_human_like_squad(
    samplers: tuple[_PositionSubsetSampler, ...],
    combined_cost_weights: np.ndarray,
    rules: SquadRules,
    rng: np.random.Generator,
) -> tuple[int, ...] | None:
    """Draw from the exact ownership-product distribution, then enforce clubs."""
    final_cost_weights = combined_cost_weights[-1]
    total_cost = _weighted_choice(final_cost_weights, rng)
    remaining_cost = total_cost
    position_costs = [0] * len(samplers)
    for index in range(len(samplers), 0, -1):
        distribution = samplers[index - 1].cost_distribution
        earlier = combined_cost_weights[index - 1]
        maximum = min(remaining_cost, len(distribution) - 1)
        candidates = np.arange(maximum + 1)
        weights = distribution[candidates] * earlier[remaining_cost - candidates]
        chosen_cost = int(candidates[_weighted_choice(weights, rng)])
        position_costs[index - 1] = chosen_cost
        remaining_cost -= chosen_cost

    selected_rows = []
    for sampler, cost in zip(samplers, position_costs, strict=True):
        selected_rows.append(_sample_position_subset(sampler, cost, rng))
    selected = pd.concat(selected_rows, ignore_index=True)
    if selected["club_id"].value_counts().max() > rules.max_players_per_club:
        return None
    return tuple(selected["player_id"].astype(int))


def _sample_position_subset(
    sampler: _PositionSubsetSampler,
    target_cost: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample one fixed-count subset conditional on its exact total price."""
    chosen: list[int] = []
    remaining_count = sampler.quota
    remaining_cost = target_cost
    for item_index in range(len(sampler.players), 0, -1):
        if remaining_count == 0:
            break
        cost = int(sampler.costs[item_index - 1])
        denominator = sampler.prefix_weights[
            item_index, remaining_count, remaining_cost
        ]
        include_weight = 0.0
        if cost <= remaining_cost:
            include_weight = (
                sampler.weights[item_index - 1]
                * sampler.prefix_weights[
                    item_index - 1,
                    remaining_count - 1,
                    remaining_cost - cost,
                ]
            )
        probability = include_weight / denominator if denominator > 0 else 0.0
        if rng.random() < min(max(probability, 0.0), 1.0):
            chosen.append(item_index - 1)
            remaining_count -= 1
            remaining_cost -= cost
    if remaining_count != 0 or remaining_cost != 0:
        raise RuntimeError("Could not sample an ownership-weighted player subset")
    return sampler.players.iloc[chosen]


def _weighted_choice(weights: np.ndarray, rng: np.random.Generator) -> int:
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Cannot sample from an empty reference distribution")
    return int(rng.choice(len(weights), p=weights / total))


def _price_ticks(value: object) -> int:
    numeric = float(value)
    ticks = round(numeric * PRICE_SCALE)
    if not np.isfinite(numeric) or not np.isclose(
        numeric * PRICE_SCALE, ticks, atol=1e-7
    ):
        raise ValueError(f"Player price {value!r} is not in 0.1m increments")
    return int(ticks)


def _reference_projection(
    selected: pd.DataFrame,
    selection: SquadSelection,
    rules: SquadRules,
    horizon: int,
) -> dict[str, object]:
    """Aggregate a squad already made legal by the constrained draw."""
    indexed = selected.set_index("player_id")
    xi = indexed.loc[list(selection.starting_xi)]
    bench = indexed.loc[list(selection.bench)]
    column = "availability_adjusted_points"
    by_position = xi.assign(
        _position=xi["position_short_name"].astype(str).str.upper()
    ).groupby("_position")[column].sum()
    counts = xi["position_id"].value_counts().to_dict()
    outfield_counts = [
        int(counts.get(position.position_id, 0))
        for position in rules.positions
        if position.short_name.upper() != "GKP"
    ]
    starting_points = float(xi[column].sum())
    captain_points = float(indexed.loc[selection.captain, column])
    return {
        "horizon": horizon,
        "formation": "-".join(str(value) for value in outfield_counts),
        "goalkeeper_points": float(by_position.get("GKP", 0.0)),
        "defence_points": float(by_position.get("DEF", 0.0)),
        "midfield_points": float(by_position.get("MID", 0.0)),
        "forward_points": float(by_position.get("FWD", 0.0)),
        "bench_points": float(bench[column].sum()),
        "captain_points": captain_points,
        "starting_xi_points": starting_points,
        "overall_points": starting_points + captain_points,
    }


def _draw_squad(
    players: pd.DataFrame,
    rules: SquadRules,
    strategy: str,
    rng: np.random.Generator,
) -> tuple[int, ...] | None:
    selected_parts = []
    for position in rules.positions:
        pool = players[players["position_id"] == position.position_id]
        if len(pool) < position.squad_count:
            return None
        weights = _selection_weights(pool, strategy)
        indices = rng.choice(
            len(pool),
            size=position.squad_count,
            replace=False,
            p=weights / weights.sum(),
        )
        selected_parts.append(pool.iloc[indices])
    selected = pd.concat(selected_parts, ignore_index=True)
    if selected["club_id"].value_counts().max() > rules.max_players_per_club:
        return None
    cost = float(pd.to_numeric(selected["price"], errors="coerce").sum())
    if not np.isfinite(cost) or cost > rules.budget + 1e-9:
        return None
    if strategy == "price_aware" and cost < rules.budget * 0.85:
        return None
    return tuple(selected["player_id"].astype(int))


def _selection_weights(players: pd.DataFrame, strategy: str) -> np.ndarray:
    if strategy == "broad_legal":
        return np.ones(len(players), dtype=float)
    if strategy == "price_aware":
        prices = pd.to_numeric(players["price"], errors="coerce").fillna(0)
        return np.maximum(prices.to_numpy(dtype=float), 0.1) ** 2
    ownership = pd.to_numeric(
        players.get("ownership_percent", pd.Series(0, index=players.index)),
        errors="coerce",
    ).fillna(0)
    return np.maximum(ownership.to_numpy(dtype=float), 0) + 1.0


def _best_selection(
    squad_players: pd.DataFrame,
    rules: SquadRules,
) -> SquadSelection:
    point_column = "availability_adjusted_points"
    candidates: list[tuple[float, tuple[int, ...]]] = []
    position_pools = {
        position.position_id: squad_players[
            squad_players["position_id"] == position.position_id
        ].sort_values(
            [point_column, "player_id"],
            ascending=[False, True],
        )
        for position in rules.positions
    }
    ranges = [
        range(position.minimum_starters, position.maximum_starters + 1)
        for position in rules.positions
    ]
    for counts in product(*ranges):
        if sum(counts) != rules.starting_size:
            continue
        starters = []
        points = 0.0
        for position, count in zip(rules.positions, counts, strict=True):
            chosen = position_pools[position.position_id].head(count)
            starters.extend(chosen["player_id"].astype(int))
            points += float(chosen[point_column].sum())
        starter_ids = tuple(starters)
        candidates.append((points, starter_ids))
    if not candidates:
        raise ValueError("API position rules do not permit a starting formation")
    _, starting_xi = max(candidates, key=lambda value: (value[0], value[1]))
    starter_set = set(starting_xi)
    bench_rows = squad_players[
        ~squad_players["player_id"].isin(starter_set)
    ].sort_values(
        ["position_id", point_column, "player_id"],
        ascending=[True, False, True],
    )
    bench = tuple(bench_rows["player_id"].astype(int))
    ranked_starters = squad_players[
        squad_players["player_id"].isin(starter_set)
    ].sort_values([point_column, "player_id"], ascending=[False, True])
    captain, vice_captain = ranked_starters.head(2)["player_id"].astype(int)
    return SquadSelection(
        player_ids=tuple(squad_players["player_id"].astype(int)),
        starting_xi=starting_xi,
        bench=bench,
        captain=int(captain),
        vice_captain=int(vice_captain),
        source="generated",
    )
