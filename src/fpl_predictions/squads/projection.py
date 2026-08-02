"""Availability adjustment and squad-level point aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.validation import ValidatedSquad


@dataclass(frozen=True, slots=True)
class SquadProjection:
    """Projected components for one legal squad and one horizon."""

    horizon: int
    formation: str
    goalkeeper_points: float
    defence_points: float
    midfield_points: float
    forward_points: float
    bench_points: float
    captain_points: float
    starting_xi_points: float
    overall_points: float
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "formation": self.formation,
            "goalkeeper_points": self.goalkeeper_points,
            "defence_points": self.defence_points,
            "midfield_points": self.midfield_points,
            "forward_points": self.forward_points,
            "bench_points": self.bench_points,
            "captain_points": self.captain_points,
            "starting_xi_points": self.starting_xi_points,
            "overall_points": self.overall_points,
            "warnings": list(self.warnings),
        }


def add_availability_adjusted_predictions(
    players: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Adjust the first gameweek only when the API supplies availability."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    horizon_column = f"predicted_points_{horizon}"
    required = {"player_id", "status", horizon_column}
    if horizon > 1:
        required.add("predicted_points_1")
    missing = required.difference(players.columns)
    if missing:
        raise ValueError(
            "Prediction table is missing columns: "
            + ", ".join(sorted(missing))
        )

    result = players.copy()
    total = pd.to_numeric(result[horizon_column], errors="coerce")
    first = (
        pd.to_numeric(result["predicted_points_1"], errors="coerce")
        if horizon > 1
        else total
    )
    if total.isna().any() or first.isna().any():
        raise ValueError("Selected prediction columns must be numeric and non-null")

    chance = pd.Series(float("nan"), index=result.index, dtype="float64")
    for column in (
        "chance_of_playing_this_round",
        "chance_of_playing_next_round",
    ):
        if column in result:
            values = pd.to_numeric(result[column], errors="coerce")
            chance = chance.fillna(values)
    invalid = chance.notna() & ~chance.between(0, 100)
    if invalid.any():
        ids = result.loc[invalid, "player_id"].astype(int).tolist()
        raise ValueError(f"Availability percentages outside 0-100 for IDs: {ids}")

    status = result["status"].fillna("").astype(str).str.lower()
    factor = chance / 100.0
    factor = factor.where(chance.notna(), status.eq("a").astype(float))
    unknown = chance.isna() & status.ne("a")
    factor = factor.where(~unknown, 1.0)
    result["availability_factor"] = factor
    result["availability_adjusted_points"] = total - first * (1.0 - factor)

    warnings = []
    if unknown.any():
        ids = result.loc[unknown, "player_id"].astype(int).tolist()
        warnings.append(
            "No numeric availability estimate for unavailable/doubtful player "
            f"IDs {ids}; raw predictions were retained."
        )
    adjusted = factor.lt(1.0)
    if adjusted.any():
        ids = result.loc[adjusted, "player_id"].astype(int).tolist()
        warnings.append(
            "First-gameweek projections were availability-adjusted for player "
            f"IDs {ids}; later gameweeks remain unadjusted."
        )
    return result, tuple(warnings)


def project_squad(
    squad: ValidatedSquad,
    predictions: pd.DataFrame,
    rules: SquadRules,
    horizon: int,
) -> SquadProjection:
    """Aggregate starters by position, bench, and captain bonus."""
    prediction_columns = [
        column
        for column in predictions.columns
        if column not in squad.player_rows.columns or column == "player_id"
    ]
    combined = squad.player_rows.merge(
        predictions.loc[:, prediction_columns],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    horizon_column = f"predicted_points_{horizon}"
    if horizon_column not in combined:
        raise ValueError(f"Prediction table has no horizon-{horizon} predictions")
    missing = combined[horizon_column].isna()
    if missing.any():
        missing_ids = combined.loc[missing, "player_id"].astype(int).tolist()
        raise ValueError(
            f"Missing horizon-{horizon} predictions for player IDs: {missing_ids}"
        )
    adjusted, warnings = add_availability_adjusted_predictions(combined, horizon)
    indexed = adjusted.set_index("player_id")
    xi = indexed.loc[list(squad.selection.starting_xi)]
    bench = indexed.loc[list(squad.selection.bench)]
    point_column = "availability_adjusted_points"
    position_totals = xi.groupby("position_short_name")[point_column].sum()

    def position_points(short_name: str) -> float:
        return float(position_totals.get(short_name, 0.0))

    starting_points = float(xi[point_column].sum())
    captain_points = float(indexed.loc[squad.selection.captain, point_column])
    return SquadProjection(
        horizon=horizon,
        formation=squad.formation,
        goalkeeper_points=position_points(_find_position(rules, "GKP")),
        defence_points=position_points(_find_position(rules, "DEF")),
        midfield_points=position_points(_find_position(rules, "MID")),
        forward_points=position_points(_find_position(rules, "FWD")),
        bench_points=float(bench[point_column].sum()),
        captain_points=captain_points,
        starting_xi_points=starting_points,
        overall_points=starting_points + captain_points,
        warnings=warnings,
    )


def player_projection_details(
    squad: ValidatedSquad,
    predictions: pd.DataFrame,
    horizon: int,
) -> list[dict[str, Any]]:
    """Return auditable player-level points and optional expected minutes."""
    prediction_columns = [
        column
        for column in predictions.columns
        if column not in squad.player_rows.columns or column == "player_id"
    ]
    combined = squad.player_rows.merge(
        predictions.loc[:, prediction_columns],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    adjusted, _ = add_availability_adjusted_predictions(combined, horizon)
    indexed = adjusted.set_index("player_id")
    order = list(squad.selection.starting_xi) + list(squad.selection.bench)
    details = []
    for bench_order, player_id in enumerate(order):
        row = indexed.loc[player_id]
        is_starter = bench_order < len(squad.selection.starting_xi)
        item: dict[str, Any] = {
            "player_id": int(player_id),
            "display_name": str(row.get("display_name", player_id)),
            "club_name": str(row.get("club_name", "")),
            "position": str(row.get("position_short_name", "")),
            "lineup_role": "starter" if is_starter else "bench",
            "is_captain": player_id == squad.selection.captain,
            "is_vice_captain": player_id == squad.selection.vice_captain,
            "predicted_points": float(row[f"predicted_points_{horizon}"]),
            "availability_factor_first_gameweek": float(
                row["availability_factor"]
            ),
            "availability_adjusted_points": float(
                row["availability_adjusted_points"]
            ),
        }
        minutes_column = f"predicted_minutes_{horizon}"
        if minutes_column in indexed:
            expected_minutes = float(row[minutes_column])
            first_minutes = (
                float(row["predicted_minutes_1"])
                if horizon > 1 and "predicted_minutes_1" in indexed
                else expected_minutes
            )
            item["expected_minutes"] = expected_minutes
            item["availability_adjusted_expected_minutes"] = (
                expected_minutes
                - first_minutes * (1.0 - float(row["availability_factor"]))
            )
        if "appearance_probability_1" in indexed:
            item["appearance_probability_next_gameweek"] = float(
                row["appearance_probability_1"]
            )
        if "start_probability_1" in indexed:
            item["start_probability_next_gameweek"] = float(
                row["start_probability_1"]
            )
        details.append(item)
    return details


def _find_position(rules: SquadRules, expected: str) -> str:
    matches = [
        position.short_name
        for position in rules.positions
        if position.short_name.upper() == expected
    ]
    if len(matches) != 1:
        raise ValueError(f"API rules do not define exactly one {expected} position")
    return matches[0]
