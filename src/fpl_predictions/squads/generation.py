"""Deterministic generation of legal comparison squads."""

from __future__ import annotations

from itertools import product
import json

import numpy as np
import pandas as pd

from fpl_predictions.squads.projection import (
    add_availability_adjusted_predictions,
    project_squad,
)
from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.schemas import SquadSelection
from fpl_predictions.squads.validation import validate_squad

REFERENCE_STRATEGIES = {
    "broad_legal",
    "price_aware",
    "ownership_weighted",
}


def generate_reference_population(
    players: pd.DataFrame,
    predictions: pd.DataFrame,
    rules: SquadRules,
    horizon: int,
    size: int = 1000,
    strategy: str = "price_aware",
    seed: int = 2026,
) -> pd.DataFrame:
    """Generate legal squads and score their strongest predicted starting XI."""
    if size <= 0:
        raise ValueError("reference size must be positive")
    if strategy not in REFERENCE_STRATEGIES:
        raise ValueError(
            "strategy must be one of: " + ", ".join(sorted(REFERENCE_STRATEGIES))
        )
    merged = players.merge(
        predictions,
        on="player_id",
        how="inner",
        suffixes=("", "_prediction"),
        validate="one_to_one",
    )
    adjusted, _ = add_availability_adjusted_predictions(merged, horizon)
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    attempts = 0
    maximum_attempts = max(5000, size * 500)
    while len(records) < size and attempts < maximum_attempts:
        attempts += 1
        player_ids = _draw_squad(adjusted, rules, strategy, rng)
        if player_ids is None:
            continue
        selection = _best_selection(
            adjusted[adjusted["player_id"].isin(player_ids)],
            rules,
        )
        try:
            validated = validate_squad(selection, players, rules)
        except ValueError:
            continue
        projection = project_squad(validated, predictions, rules, horizon)
        record: dict[str, object] = {
            key: value
            for key, value in projection.as_dict().items()
            if key != "warnings"
        }
        record.update(
            {
                "reference_index": len(records),
                "strategy": strategy,
                "seed": seed,
                "total_cost": validated.total_cost,
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
    ranges = [
        range(position.minimum_starters, position.maximum_starters + 1)
        for position in rules.positions
    ]
    for counts in product(*ranges):
        if sum(counts) != rules.starting_size:
            continue
        starters = []
        for position, count in zip(rules.positions, counts, strict=True):
            pool = squad_players[
                squad_players["position_id"] == position.position_id
            ].sort_values(
                [point_column, "player_id"],
                ascending=[False, True],
            )
            starters.extend(pool.head(count)["player_id"].astype(int))
        starter_ids = tuple(starters)
        points = float(
            squad_players[
                squad_players["player_id"].isin(starter_ids)
            ][point_column].sum()
        )
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
