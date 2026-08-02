"""Leakage-safe rolling club strength and opponent fixture features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


STRENGTH_PRIOR_MATCHES = 5.0


class TeamStrengthError(ValueError):
    """Raised when fixture results cannot support safe strength features."""


def add_team_strength_features(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    snapshot_gameweek: int,
    horizons: Sequence[int],
    *,
    club_column: str = "club_id",
    prior_matches: float = STRENGTH_PRIOR_MATCHES,
) -> pd.DataFrame:
    """Add strength known before ``snapshot_gameweek`` and future opposition.

    Strengths are goals-per-match ratios relative to the past league average,
    shrunk toward 1.0 by ``prior_matches``. Only fixtures from earlier
    gameweeks may contribute results. A value above 1.0 means a stronger
    attack or a leakier defence, respectively.
    """
    if snapshot_gameweek <= 0:
        raise ValueError("snapshot_gameweek must be positive")
    if prior_matches <= 0:
        raise ValueError("prior_matches must be positive")
    required_players = {club_column}
    missing_players = required_players.difference(players.columns)
    if missing_players:
        raise TeamStrengthError(
            "Player data is missing club columns: "
            + ", ".join(sorted(missing_players))
        )
    required_fixtures = {
        "gameweek_id",
        "home_club_id",
        "away_club_id",
    }
    missing_fixtures = required_fixtures.difference(fixtures.columns)
    if missing_fixtures:
        raise TeamStrengthError(
            "Fixture data is missing columns: "
            + ", ".join(sorted(missing_fixtures))
        )

    normalized_horizons = tuple(int(value) for value in horizons)
    if not normalized_horizons or any(value <= 0 for value in normalized_horizons):
        raise ValueError("horizons must contain positive integers")

    result = players.copy()
    working = fixtures.copy()
    for column in ("gameweek_id", "home_club_id", "away_club_id"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    for column in ("team_h_score", "team_a_score"):
        if column not in working:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")

    ratings, venue_factors = _ratings_before_gameweek(
        working,
        snapshot_gameweek,
        prior_matches,
    )
    club_ids = pd.to_numeric(result[club_column], errors="coerce")
    result["club_attack_strength"] = club_ids.map(
        ratings["attack_strength"]
    ).fillna(1.0)
    result["club_goals_conceded_strength"] = club_ids.map(
        ratings["goals_conceded_strength"]
    ).fillna(1.0)
    result["club_strength_matches"] = club_ids.map(
        ratings["matches"]
    ).fillna(0.0)

    for horizon in normalized_horizons:
        upcoming = working.loc[
            working["gameweek_id"].ge(snapshot_gameweek)
            & working["gameweek_id"].lt(snapshot_gameweek + horizon)
        ]
        fixture_features = _upcoming_club_features(
            upcoming,
            ratings,
            venue_factors,
        )
        for base_name in (
            "opponent_attack_strength_mean",
            "opponent_goals_conceded_strength_mean",
            "attacking_fixture_factor_mean",
            "defensive_fixture_factor_mean",
        ):
            column = f"upcoming_{base_name}_{horizon}"
            result[column] = club_ids.map(fixture_features[base_name])
            no_fixture = result.get(
                f"upcoming_fixture_count_{horizon}",
                pd.Series(np.nan, index=result.index),
            ).eq(0)
            result.loc[no_fixture, column] = 0.0
    return result


def _ratings_before_gameweek(
    fixtures: pd.DataFrame,
    snapshot_gameweek: int,
    prior_matches: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    completed = fixtures.loc[
        fixtures["gameweek_id"].lt(snapshot_gameweek)
        & fixtures["team_h_score"].notna()
        & fixtures["team_a_score"].notna()
    ].copy()
    club_ids = pd.unique(
        pd.concat(
            [fixtures["home_club_id"], fixtures["away_club_id"]],
            ignore_index=True,
        ).dropna()
    )
    ratings = pd.DataFrame(index=pd.Index(club_ids, name="club_id"))
    if completed.empty:
        ratings["matches"] = 0.0
        ratings["attack_strength"] = 1.0
        ratings["goals_conceded_strength"] = 1.0
        return ratings, {"home": 1.0, "away": 1.0}

    home = completed.loc[
        :, ["home_club_id", "team_h_score", "team_a_score"]
    ].rename(
        columns={
            "home_club_id": "club_id",
            "team_h_score": "goals_for",
            "team_a_score": "goals_against",
        }
    )
    away = completed.loc[
        :, ["away_club_id", "team_a_score", "team_h_score"]
    ].rename(
        columns={
            "away_club_id": "club_id",
            "team_a_score": "goals_for",
            "team_h_score": "goals_against",
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    totals = long.groupby("club_id").agg(
        matches=("goals_for", "size"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )
    league_goals_per_team_match = float(long["goals_for"].mean())
    if not np.isfinite(league_goals_per_team_match) or league_goals_per_team_match <= 0:
        league_goals_per_team_match = 1.0
    ratings = ratings.join(totals, how="left").fillna(0.0)
    denominator = ratings["matches"] + prior_matches
    prior_goals = prior_matches * league_goals_per_team_match
    ratings["attack_strength"] = (
        (ratings["goals_for"] + prior_goals)
        / denominator
        / league_goals_per_team_match
    )
    ratings["goals_conceded_strength"] = (
        (ratings["goals_against"] + prior_goals)
        / denominator
        / league_goals_per_team_match
    )

    home_mean = float(completed["team_h_score"].mean())
    away_mean = float(completed["team_a_score"].mean())
    return ratings, {
        "home": _safe_ratio(home_mean, league_goals_per_team_match),
        "away": _safe_ratio(away_mean, league_goals_per_team_match),
    }


def _upcoming_club_features(
    fixtures: pd.DataFrame,
    ratings: pd.DataFrame,
    venue_factors: dict[str, float],
) -> pd.DataFrame:
    columns = [
        "opponent_attack_strength_mean",
        "opponent_goals_conceded_strength_mean",
        "attacking_fixture_factor_mean",
        "defensive_fixture_factor_mean",
    ]
    if fixtures.empty:
        return pd.DataFrame(columns=columns, dtype=float)

    def strength(club_ids: pd.Series, column: str) -> pd.Series:
        return club_ids.map(ratings[column]).fillna(1.0)

    home = pd.DataFrame(
        {
            "club_id": fixtures["home_club_id"],
            "opponent_attack_strength": strength(
                fixtures["away_club_id"], "attack_strength"
            ),
            "opponent_goals_conceded_strength": strength(
                fixtures["away_club_id"], "goals_conceded_strength"
            ),
        }
    )
    home["attacking_fixture_factor"] = (
        home["opponent_goals_conceded_strength"] * venue_factors["home"]
    )
    home["defensive_fixture_factor"] = (
        home["opponent_attack_strength"] * venue_factors["away"]
    )
    away = pd.DataFrame(
        {
            "club_id": fixtures["away_club_id"],
            "opponent_attack_strength": strength(
                fixtures["home_club_id"], "attack_strength"
            ),
            "opponent_goals_conceded_strength": strength(
                fixtures["home_club_id"], "goals_conceded_strength"
            ),
        }
    )
    away["attacking_fixture_factor"] = (
        away["opponent_goals_conceded_strength"] * venue_factors["away"]
    )
    away["defensive_fixture_factor"] = (
        away["opponent_attack_strength"] * venue_factors["home"]
    )
    long = pd.concat([home, away], ignore_index=True)
    return long.groupby("club_id").agg(
        opponent_attack_strength_mean=("opponent_attack_strength", "mean"),
        opponent_goals_conceded_strength_mean=(
            "opponent_goals_conceded_strength",
            "mean",
        ),
        attacking_fixture_factor_mean=("attacking_fixture_factor", "mean"),
        defensive_fixture_factor_mean=("defensive_fixture_factor", "mean"),
    )


def _safe_ratio(value: float, denominator: float) -> float:
    if not np.isfinite(value) or not np.isfinite(denominator) or denominator <= 0:
        return 1.0
    return value / denominator
