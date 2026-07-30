"""Percentile ratings and evidence-based squad explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fpl_predictions.squads.projection import SquadProjection

RATING_COMPONENTS = {
    "goalkeeper": "goalkeeper_points",
    "defence": "defence_points",
    "midfield": "midfield_points",
    "attack": "forward_points",
    "bench": "bench_points",
    "captaincy": "captain_points",
    "overall": "overall_points",
}


@dataclass(frozen=True, slots=True)
class SquadRating:
    """A complete 0-100 rating against a named reference population."""

    scores: dict[str, float]
    reference_name: str
    reference_size: int
    interpretation: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    uncertainties: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "reference_population": {
                "name": self.reference_name,
                "size": self.reference_size,
            },
            "interpretation": self.interpretation,
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
            "uncertainties": list(self.uncertainties),
        }


def midrank_percentile(value: float, population: pd.Series) -> float:
    """Return a 0-100 empirical percentile, assigning ties their midrank."""
    values = pd.to_numeric(population, errors="coerce").dropna().to_numpy()
    if not np.isfinite(value):
        raise ValueError("rating value must be finite")
    if len(values) == 0:
        raise ValueError("reference population has no numeric values")
    less = int(np.sum(values < value))
    equal = int(np.sum(values == value))
    return 100.0 * (less + 0.5 * equal) / len(values)


def rate_squad(
    projection: SquadProjection,
    references: pd.DataFrame,
    reference_name: str,
) -> SquadRating:
    """Rate every component directly against its corresponding distribution."""
    missing = set(RATING_COMPONENTS.values()).difference(references.columns)
    if missing:
        raise ValueError(
            "Reference population is missing columns: "
            + ", ".join(sorted(missing))
        )
    projection_values = projection.as_dict()
    scores = {
        name: round(
            midrank_percentile(
                float(projection_values[column]),
                references[column],
            ),
            1,
        )
        for name, column in RATING_COMPONENTS.items()
    }
    strengths = tuple(
        name for name, score in scores.items() if score >= 75.0
    )
    weaknesses = tuple(
        name for name, score in scores.items() if score <= 25.0
    )
    overall = scores["overall"]
    interpretation = (
        f"Overall projection ranks at the {overall:.1f}th percentile "
        f"of {len(references)} {reference_name} squads. Overall is calculated "
        "directly from projected XI points plus the captain bonus; it is not "
        "an average of the component scores."
    )
    uncertainties = tuple(projection.warnings) + (
        "Ratings are relative to a simulated reference population and inherit "
        "the uncertainty and omissions of the player-point model.",
        "Bench points are reported separately and are not included in overall "
        "unless a future chip- or substitution-aware simulation is used.",
    )
    return SquadRating(
        scores=scores,
        reference_name=reference_name,
        reference_size=len(references),
        interpretation=interpretation,
        strengths=strengths,
        weaknesses=weaknesses,
        uncertainties=uncertainties,
    )
