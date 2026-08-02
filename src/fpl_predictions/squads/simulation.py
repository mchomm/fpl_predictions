"""Reproducible appearance, automatic-substitution, and captain simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fpl_predictions.squads.projection import SquadProjection
from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.schemas import SquadSelection


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Mean projection plus diagnostics from availability scenarios."""

    projection: SquadProjection
    simulations: int
    seed: int
    expected_autosub_points: float
    expected_vice_captain_points: float
    autosub_probability: float
    vice_captain_takeover_probability: float
    expected_unfilled_starting_slots: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "independent Bernoulli appearance Monte Carlo",
            "simulations": self.simulations,
            "seed": self.seed,
            "expected_autosub_points": self.expected_autosub_points,
            "expected_vice_captain_points": self.expected_vice_captain_points,
            "autosub_probability": self.autosub_probability,
            "vice_captain_takeover_probability": (
                self.vice_captain_takeover_probability
            ),
            "expected_unfilled_starting_slots": (
                self.expected_unfilled_starting_slots
            ),
            "projection": self.projection.as_dict(),
            "assumptions": [
                "Player appearance events are independent within a gameweek.",
                "The calibrated next-gameweek probability is reused for later "
                "gameweeks when no later-week probability artifact exists.",
                "Projected points are converted to appearance-conditional points "
                "so absence is not counted twice.",
                "Chips are not simulated.",
            ],
        }


def simulate_selection(
    players: pd.DataFrame,
    selection: SquadSelection,
    rules: SquadRules,
    horizon: int,
    simulations: int = 500,
    seed: int = 2026,
) -> SimulationResult:
    """Simulate FPL autosubs and captain fallback for one legal selection."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    required = {
        "player_id",
        "position_id",
        "position_short_name",
        f"predicted_points_{horizon}",
        "appearance_probability_1",
    }
    if horizon > 1:
        required.add("predicted_points_1")
    missing = required.difference(players.columns)
    if missing:
        raise ValueError(
            "Simulation requires prediction columns: "
            + ", ".join(sorted(missing))
        )
    indexed = players.drop_duplicates("player_id").set_index("player_id")
    missing_ids = set(selection.player_ids).difference(indexed.index)
    if missing_ids:
        raise ValueError(f"Simulation is missing player IDs: {sorted(missing_ids)}")
    order = list(selection.player_ids)
    squad = indexed.loc[order].copy()
    id_to_index = {player_id: index for index, player_id in enumerate(order)}
    xi_indices = [id_to_index[value] for value in selection.starting_xi]
    bench_indices = [id_to_index[value] for value in selection.bench]
    captain_index = id_to_index[selection.captain]
    vice_index = id_to_index[selection.vice_captain]

    unconditional = _per_gameweek_points(squad, horizon)
    base_probability = pd.to_numeric(
        squad["appearance_probability_1"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(base_probability).all():
        raise ValueError("Appearance probabilities must be numeric and non-null")
    base_probability = np.clip(base_probability, 0.0, 1.0)
    probabilities = np.repeat(base_probability[None, :], horizon, axis=0)
    probabilities[0], warnings = _api_adjusted_first_probability(
        squad,
        probabilities[0],
    )
    if "recent_gameweeks_available" in squad:
        history = pd.to_numeric(
            squad["recent_gameweeks_available"], errors="coerce"
        )
        if history.fillna(0).median() < 2:
            warnings.append(
                "Participation probabilities are cold-start estimates because "
                "the squad has fewer than two recent live gameweeks."
            )
    conditional = np.divide(
        unconditional,
        base_probability[None, :],
        out=np.zeros_like(unconditional),
        where=base_probability[None, :] > 1e-9,
    )

    rng = np.random.default_rng(seed)
    appeared = rng.random((simulations, horizon, len(squad))) < probabilities
    position_ids = squad["position_id"].astype(int).to_numpy()
    position_names = squad["position_short_name"].astype(str).str.upper().to_numpy()
    goalkeeper_ids = {
        position.position_id
        for position in rules.positions
        if position.short_name.upper() == "GKP"
    }
    if len(goalkeeper_ids) != 1:
        raise ValueError("Simulation requires exactly one goalkeeper position")
    goalkeeper_id = next(iter(goalkeeper_ids))
    rule_by_id = rules.position_by_id

    component_sums = {name: 0.0 for name in ("GKP", "DEF", "MID", "FWD")}
    fielded_sum = 0.0
    captain_sum = 0.0
    autosub_sum = 0.0
    vice_sum = 0.0
    autosub_events = 0
    vice_events = 0
    unfilled_sum = 0.0
    original_counts = {
        position_id: sum(position_ids[index] == position_id for index in xi_indices)
        for position_id in rule_by_id
    }

    for simulation_index in range(simulations):
        for gameweek_index in range(horizon):
            availability = appeared[simulation_index, gameweek_index]
            scoring = [index for index in xi_indices if availability[index]]
            absent_slots = [index for index in xi_indices if not availability[index]]
            replacement_slots: dict[int, int] = {}

            absent_goalkeepers = [
                index
                for index in absent_slots
                if position_ids[index] == goalkeeper_id
            ]
            bench_goalkeepers = [
                index
                for index in bench_indices
                if position_ids[index] == goalkeeper_id and availability[index]
            ]
            if absent_goalkeepers and bench_goalkeepers:
                replacement_slots[absent_goalkeepers[0]] = bench_goalkeepers[0]
                scoring.append(bench_goalkeepers[0])

            for candidate in bench_indices:
                if position_ids[candidate] == goalkeeper_id or not availability[candidate]:
                    continue
                chosen_slot = _legal_replacement_slot(
                    candidate,
                    absent_slots,
                    replacement_slots,
                    position_ids,
                    original_counts,
                    rule_by_id,
                    goalkeeper_id,
                )
                if chosen_slot is None:
                    continue
                replacement_slots[chosen_slot] = candidate
                scoring.append(candidate)

            gameweek_points = sum(
                conditional[gameweek_index, index] for index in scoring
            )
            fielded_sum += gameweek_points
            for index in scoring:
                name = position_names[index]
                if name in component_sums:
                    component_sums[name] += conditional[gameweek_index, index]

            substitute_indices = set(replacement_slots.values())
            autosub_points = sum(
                conditional[gameweek_index, index]
                for index in substitute_indices
            )
            autosub_sum += autosub_points
            autosub_events += bool(substitute_indices)
            unfilled_sum += len(absent_slots) - len(replacement_slots)

            if availability[captain_index]:
                captain_sum += conditional[gameweek_index, captain_index]
            elif availability[vice_index]:
                fallback = conditional[gameweek_index, vice_index]
                captain_sum += fallback
                vice_sum += fallback
                vice_events += 1

    denominator = float(simulations)
    fielded_mean = fielded_sum / denominator
    captain_mean = captain_sum / denominator
    bench_raw = float(
        unconditional[:, bench_indices].sum()
    )
    formation = _formation(selection, indexed, rules)
    projection = SquadProjection(
        horizon=horizon,
        formation=formation,
        goalkeeper_points=component_sums["GKP"] / denominator,
        defence_points=component_sums["DEF"] / denominator,
        midfield_points=component_sums["MID"] / denominator,
        forward_points=component_sums["FWD"] / denominator,
        bench_points=bench_raw,
        captain_points=captain_mean,
        starting_xi_points=fielded_mean,
        overall_points=fielded_mean + captain_mean,
        warnings=tuple(warnings)
        + (
            "Overall uses appearance scenarios, legal automatic substitutions, "
            "and vice-captain takeover.",
        ),
    )
    scenario_count = float(simulations * horizon)
    return SimulationResult(
        projection=projection,
        simulations=simulations,
        seed=seed,
        expected_autosub_points=autosub_sum / denominator,
        expected_vice_captain_points=vice_sum / denominator,
        autosub_probability=autosub_events / scenario_count,
        vice_captain_takeover_probability=vice_events / scenario_count,
        expected_unfilled_starting_slots=unfilled_sum / scenario_count,
    )


def _per_gameweek_points(players: pd.DataFrame, horizon: int) -> np.ndarray:
    cumulative_horizons = sorted(
        {
            int(column.removeprefix("predicted_points_"))
            for column in players.columns
            if column.startswith("predicted_points_")
            and column.removeprefix("predicted_points_").isdigit()
            and int(column.removeprefix("predicted_points_")) <= horizon
        }
        | {horizon}
    )
    points = np.zeros((horizon, len(players)), dtype=float)
    previous_horizon = 0
    previous_total = np.zeros(len(players), dtype=float)
    for current_horizon in cumulative_horizons:
        column = f"predicted_points_{current_horizon}"
        if column not in players:
            continue
        total = pd.to_numeric(players[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(total).all():
            raise ValueError(f"{column} must be numeric and non-null")
        steps = current_horizon - previous_horizon
        if steps <= 0:
            continue
        marginal = np.maximum(total - previous_total, 0.0) / steps
        points[previous_horizon:current_horizon] = marginal
        previous_horizon = current_horizon
        previous_total = total
    if previous_horizon != horizon:
        raise ValueError(
            f"Cannot distribute points through horizon {horizon}; "
            f"missing predicted_points_{horizon}"
        )
    return points


def _api_adjusted_first_probability(
    players: pd.DataFrame,
    base: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    adjusted = base.copy()
    chance = pd.Series(np.nan, index=players.index, dtype=float)
    for column in (
        "chance_of_playing_this_round",
        "chance_of_playing_next_round",
    ):
        if column in players:
            chance = chance.fillna(pd.to_numeric(players[column], errors="coerce"))
    invalid = chance.notna() & ~chance.between(0, 100)
    if invalid.any():
        raise ValueError("Availability percentages must be between 0 and 100")
    numeric = chance.notna().to_numpy()
    adjusted[numeric] = np.minimum(
        adjusted[numeric], chance.to_numpy(dtype=float)[numeric] / 100.0
    )
    status = players.get("status", pd.Series("a", index=players.index))
    unknown = chance.isna() & status.fillna("").astype(str).str.lower().ne("a")
    warnings = []
    if unknown.any():
        ids = players.loc[unknown, "player_id"].astype(int).tolist()
        warnings.append(
            "No numeric current availability estimate for player IDs "
            f"{ids}; calibrated model probabilities were retained."
        )
    return adjusted, warnings


def _legal_replacement_slot(
    candidate: int,
    absent_slots: list[int],
    replacements: dict[int, int],
    position_ids: np.ndarray,
    original_counts: dict[int, int],
    rule_by_id: dict[int, Any],
    goalkeeper_id: int,
) -> int | None:
    for slot in absent_slots:
        if slot in replacements or position_ids[slot] == goalkeeper_id:
            continue
        counts = dict(original_counts)
        for replaced_slot, replacement in replacements.items():
            counts[position_ids[replaced_slot]] -= 1
            counts[position_ids[replacement]] += 1
        counts[position_ids[slot]] -= 1
        counts[position_ids[candidate]] += 1
        if all(
            rule.minimum_starters <= counts.get(position_id, 0) <= rule.maximum_starters
            for position_id, rule in rule_by_id.items()
        ):
            return slot
    return None


def _formation(
    selection: SquadSelection,
    indexed: pd.DataFrame,
    rules: SquadRules,
) -> str:
    xi = indexed.loc[list(selection.starting_xi)]
    counts = xi["position_id"].value_counts().to_dict()
    outfield = [
        str(int(counts.get(position.position_id, 0)))
        for position in rules.positions
        if position.short_name.upper() != "GKP"
    ]
    return "-".join(outfield)
