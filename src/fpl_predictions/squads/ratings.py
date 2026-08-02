"""Percentile ratings and evidence-based squad explanations."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
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

SCHOOL_SCORE_CENTER = 75.0
SCHOOL_SCORE_SPREAD = 8.0
SCHOOL_SCORE_MAXIMUM = 99.9


@dataclass(frozen=True, slots=True)
class SquadRating:
    """A complete 0-100 rating against a named reference population."""

    scores: dict[str, float]
    percentiles: dict[str, float]
    reference_name: str
    reference_size: int
    interpretation: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    uncertainties: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "percentiles": self.percentiles,
            "score_calibration": {
                "name": "school_style_v1",
                "formula": (
                    "75 + 8 * inverse_normal_cdf(smoothed_midrank_probability)"
                ),
                "median_reference_score": SCHOOL_SCORE_CENTER,
                "score_points_per_standard_deviation": SCHOOL_SCORE_SPREAD,
                "maximum_score": SCHOOL_SCORE_MAXIMUM,
                "notes": (
                    "Monotonic presentation transform only; projections and "
                    "comparative ordering are unchanged."
                ),
            },
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


def school_style_score(value: float, population: pd.Series) -> float:
    """Map comparative rank to an intuitive school-style score."""
    values = pd.to_numeric(population, errors="coerce").dropna().to_numpy()
    if not np.isfinite(value):
        raise ValueError("rating value must be finite")
    if len(values) == 0:
        raise ValueError("reference population has no numeric values")
    less = int(np.sum(values < value))
    equal = int(np.sum(values == value))
    probability = (less + 0.5 * equal + 0.5) / (len(values) + 1.0)
    score = SCHOOL_SCORE_CENTER + SCHOOL_SCORE_SPREAD * NormalDist().inv_cdf(
        probability
    )
    return min(max(score, 0.0), SCHOOL_SCORE_MAXIMUM)


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
    percentiles = {
        name: round(
            midrank_percentile(
                float(projection_values[column]),
                references[column],
            ),
            1,
        )
        for name, column in RATING_COMPONENTS.items()
    }
    scores = {
        name: round(
            school_style_score(
                float(projection_values[column]),
                references[column],
            ),
            1,
        )
        for name, column in RATING_COMPONENTS.items()
    }
    strengths = tuple(
        name for name, score in scores.items() if score >= 85.0
    )
    weaknesses = tuple(
        name for name, score in scores.items() if score <= 65.0
    )
    overall = scores["overall"]
    overall_percentile = percentiles["overall"]
    interpretation = (
        f"Overall score is {overall:.1f}/100; the underlying projection ranks "
        f"at the {overall_percentile:.1f}th percentile of the "
        f"{len(references)}-squad {reference_name}. A median reference squad "
        "scores 75. Overall is calculated directly from projected XI points "
        "plus the captain bonus; it is not an average of component scores."
    )
    substitution_simulated = any(
        "automatic substitutions" in warning.lower()
        for warning in projection.warnings
    )
    substitution_note = (
        "Automatic substitutions and vice-captain takeover use calibrated, "
        "independent appearance scenarios; correlated team news is not modeled."
        if substitution_simulated
        else "Automatic substitutions and vice-captain takeover are not simulated."
    )
    uncertainties = tuple(projection.warnings) + (
        "Ratings are relative to a simulated reference population and inherit "
        "the uncertainty and omissions of the player-point model.",
        "Bench points are reported separately; when scenario simulation is "
        "enabled, only legal automatic-substitution contributions enter overall.",
        substitution_note,
        "The school-style score is a monotonic display calibration; consult the "
        "saved percentile and raw projected points for the underlying evidence.",
    )
    return SquadRating(
        scores=scores,
        percentiles=percentiles,
        reference_name=reference_name,
        reference_size=len(references),
        interpretation=interpretation,
        strengths=strengths,
        weaknesses=weaknesses,
        uncertainties=uncertainties,
    )
