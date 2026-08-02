"""Tests for leakage-safe club and opponent strength features."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_predictions.data.team_strength import add_team_strength_features


def _players() -> pd.DataFrame:
    return pd.DataFrame({"player_id": [1, 2, 3], "club_id": [10, 20, 30]})


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gameweek_id": 1,
                "home_club_id": 10,
                "away_club_id": 20,
                "team_h_score": 4,
                "team_a_score": 0,
            },
            {
                "gameweek_id": 2,
                "home_club_id": 20,
                "away_club_id": 10,
                "team_h_score": 1,
                "team_a_score": 2,
            },
            {
                "gameweek_id": 3,
                "home_club_id": 10,
                "away_club_id": 30,
                "team_h_score": pd.NA,
                "team_a_score": pd.NA,
            },
            {
                "gameweek_id": 3,
                "home_club_id": 20,
                "away_club_id": 10,
                "team_h_score": pd.NA,
                "team_a_score": pd.NA,
            },
        ]
    )


def test_strength_uses_only_prior_results_and_shrinks_small_samples() -> None:
    result = add_team_strength_features(
        _players(), _fixtures(), snapshot_gameweek=3, horizons=(1,)
    ).set_index("club_id")

    assert result.loc[10, "club_strength_matches"] == 2
    assert result.loc[10, "club_attack_strength"] > 1
    assert result.loc[10, "club_goals_conceded_strength"] < 1
    assert result.loc[20, "club_attack_strength"] < 1
    assert result.loc[30, "club_attack_strength"] == 1
    assert result.loc[30, "club_strength_matches"] == 0
    assert result.loc[30, "upcoming_opponent_attack_strength_mean_1"] > 1


def test_future_scores_cannot_change_snapshot_strength() -> None:
    fixtures = _fixtures()
    baseline = add_team_strength_features(
        _players(), fixtures, snapshot_gameweek=3, horizons=(1,)
    )
    changed = fixtures.copy()
    changed.loc[changed["gameweek_id"] == 3, "team_h_score"] = 25
    changed.loc[changed["gameweek_id"] == 3, "team_a_score"] = 17
    rerun = add_team_strength_features(
        _players(), changed, snapshot_gameweek=3, horizons=(1,)
    )

    strength_columns = [
        column
        for column in baseline
        if "strength" in column or "fixture_factor" in column
    ]
    pd.testing.assert_frame_equal(
        baseline[strength_columns], rerun[strength_columns]
    )


def test_blank_horizon_is_explicitly_zero_not_neutral() -> None:
    result = _players().assign(upcoming_fixture_count_1=[0, 0, 0])
    enriched = add_team_strength_features(
        result,
        _fixtures().loc[lambda frame: frame["gameweek_id"] < 3],
        snapshot_gameweek=3,
        horizons=(1,),
    )

    assert enriched["upcoming_attacking_fixture_factor_mean_1"].eq(0).all()
    assert enriched["upcoming_defensive_fixture_factor_mean_1"].eq(0).all()


def test_early_season_neutral_prior_is_data_independent() -> None:
    result = add_team_strength_features(
        _players(), _fixtures(), snapshot_gameweek=1, horizons=(1,)
    ).set_index("club_id")

    assert result.loc[10, "club_attack_strength"] == pytest.approx(1.0)
    assert result.loc[20, "club_goals_conceded_strength"] == pytest.approx(1.0)
