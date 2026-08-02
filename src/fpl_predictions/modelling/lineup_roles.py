"""Joint, all-club reconciliation of independently predicted lineup roles."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reconcile_lineup_probabilities(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Project player probabilities onto one GK and ten outfield starts per club.

    The adjustment preserves each model's ordering by applying one shared odds
    shift within each club/role group. It does not contain player or club names.
    """
    required = {
        "player_id",
        "club_id",
        "position_short_name",
        "appearance_probability_1",
        "start_probability_1",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            "Lineup-role reconciliation is missing columns: "
            + ", ".join(sorted(missing))
        )
    result = predictions.copy()
    appearance = pd.to_numeric(
        result["appearance_probability_1"], errors="coerce"
    )
    starts = pd.to_numeric(result["start_probability_1"], errors="coerce")
    if appearance.isna().any() or starts.isna().any():
        raise ValueError("Lineup-role probabilities must be numeric and non-null")
    if not appearance.between(0, 1).all() or not starts.between(0, 1).all():
        raise ValueError("Lineup-role probabilities must be between zero and one")

    result["independent_appearance_probability_1"] = appearance
    result["independent_start_probability_1"] = starts
    reconciled = starts.to_numpy(dtype=float).copy()
    club_audit: list[dict[str, object]] = []
    positions = result["position_short_name"].fillna("").astype(str).str.upper()
    for club_id, club in result.groupby("club_id", sort=True):
        goalkeeper_indexes = club.index[positions.loc[club.index].eq("GKP")]
        outfield_indexes = club.index[~positions.loc[club.index].eq("GKP")]
        before = float(starts.loc[club.index].sum())
        group_records = []
        for label, indexes, target in (
            ("goalkeeper", goalkeeper_indexes, 1.0),
            ("outfield", outfield_indexes, 10.0),
        ):
            if len(indexes) == 0:
                continue
            feasible_target = min(target, float(len(indexes)))
            values = starts.loc[indexes].to_numpy(dtype=float)
            adjusted = _shared_odds_projection(values, feasible_target)
            reconciled[indexes.to_numpy()] = adjusted
            group_records.append(
                {
                    "group": label,
                    "players": int(len(indexes)),
                    "target": feasible_target,
                    "before": float(values.sum()),
                    "after": float(adjusted.sum()),
                }
            )
        club_audit.append(
            {
                "club_id": int(club_id),
                "expected_starters_before": before,
                "expected_starters_after": float(reconciled[club.index].sum()),
                "groups": group_records,
            }
        )

    result["start_probability_1"] = reconciled
    result["appearance_probability_1"] = np.maximum(
        appearance.to_numpy(dtype=float), reconciled
    )
    audit: dict[str, object] = {
        "method": "shared log-odds projection by club and goalkeeper/outfield role",
        "constraints": [
            "each club's goalkeeper start probabilities sum to 1",
            "each club's outfield start probabilities sum to 10",
            "appearance probability is at least start probability",
        ],
        "clubs": club_audit,
        "player_or_club_overrides": 0,
    }
    return result, audit


def _shared_odds_projection(
    probabilities: np.ndarray,
    target: float,
) -> np.ndarray:
    """Find a shared log-odds shift whose adjusted values sum to target."""
    count = len(probabilities)
    if target <= 0:
        return np.zeros(count, dtype=float)
    if target >= count:
        return np.ones(count, dtype=float)
    epsilon = 1e-9
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped))
    lower, upper = -40.0, 40.0
    for _ in range(100):
        middle = (lower + upper) / 2.0
        total = float((1.0 / (1.0 + np.exp(-(logits + middle)))).sum())
        if total < target:
            lower = middle
        else:
            upper = middle
    shift = (lower + upper) / 2.0
    return 1.0 / (1.0 + np.exp(-(logits + shift)))
