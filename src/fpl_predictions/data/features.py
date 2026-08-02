"""Leakage-safe feature and future-label construction."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from fpl_predictions.data.team_strength import add_team_strength_features


class FeatureValidationError(ValueError):
    """Raised when source data cannot support leakage-safe training rows."""


def build_training_table(
    player_snapshots: pd.DataFrame,
    player_gameweek_stats: pd.DataFrame,
    fixture_snapshots: pd.DataFrame | None = None,
    horizons: Sequence[int] = (1, 3, 5),
    recent_window: int = 3,
) -> pd.DataFrame:
    """Build prediction-time rows with past-only features and future labels.

    A snapshot for gameweek ``G`` may use finalized stats only from gameweeks
    below ``G``. Its horizon labels begin at ``G``. Labels remain null until all
    gameweeks in that horizon are present in the finalized stats source.
    """
    if player_gameweek_stats.empty:
        raise FeatureValidationError(
            "Player gameweek stats contain no finalized rows"
        )
    normalized_horizons = _validate_horizons(horizons)
    features = build_prediction_table(
        player_snapshots,
        player_gameweek_stats,
        fixture_snapshots,
        horizons=normalized_horizons,
        recent_window=recent_window,
    )
    stats = player_gameweek_stats.copy()
    for column in ("gameweek", "player_id", "total_points", "minutes"):
        stats[column] = pd.to_numeric(stats[column], errors="raise")
    stats["_appeared"] = stats["minutes"].gt(0).astype(float)
    if "starts" in stats:
        starts = pd.to_numeric(stats["starts"], errors="coerce")
        stats["_started"] = starts.gt(0).astype(float)
    else:
        stats["_started"] = stats["minutes"].ge(60).astype(float)
    labeled_groups: list[pd.DataFrame] = []
    group_columns = ["season", "snapshot_gameweek"]
    for (season, gameweek), feature_group in features.groupby(
        group_columns, sort=True
    ):
        group = feature_group.copy()
        season_stats = stats.loc[stats["season"] == season]
        available_gameweeks = set(
            season_stats["gameweek"].dropna().astype(int).unique()
        )
        for horizon in normalized_horizons:
            label = f"label_next_{horizon}_gameweek"
            minutes_label = f"label_minutes_next_{horizon}_gameweek"
            appearance_label = f"label_appearances_next_{horizon}_gameweek"
            start_label = f"label_starts_next_{horizon}_gameweek"
            if horizon != 1:
                label += "s"
                minutes_label += "s"
                appearance_label += "s"
                start_label += "s"
            target_gameweeks = set(range(int(gameweek), int(gameweek) + horizon))
            if not target_gameweeks.issubset(available_gameweeks):
                group[label] = pd.NA
                group[minutes_label] = pd.NA
                group[appearance_label] = pd.NA
                group[start_label] = pd.NA
                continue
            future = season_stats.loc[
                season_stats["gameweek"].isin(target_gameweeks)
            ]
            labels = future.groupby("player_id", as_index=False).agg(
                **{
                    label: ("total_points", "sum"),
                    minutes_label: ("minutes", "sum"),
                    appearance_label: ("_appeared", "sum"),
                    start_label: ("_started", "sum"),
                },
                _label_gameweeks=("gameweek", "nunique"),
            )
            labels.loc[
                labels["_label_gameweeks"] != horizon, label
            ] = pd.NA
            labels.loc[
                labels["_label_gameweeks"] != horizon, minutes_label
            ] = pd.NA
            labels.loc[
                labels["_label_gameweeks"] != horizon, appearance_label
            ] = pd.NA
            labels.loc[
                labels["_label_gameweeks"] != horizon, start_label
            ] = pd.NA
            group = group.merge(
                labels[
                    [
                        "player_id",
                        label,
                        minutes_label,
                        appearance_label,
                        start_label,
                    ]
                ],
                on="player_id",
                how="left",
                validate="one_to_one",
            )
        labeled_groups.append(group)

    return pd.concat(labeled_groups, ignore_index=True).sort_values(
        ["season", "snapshot_gameweek", "player_id"], ignore_index=True
    )


def build_prediction_table(
    player_snapshots: pd.DataFrame,
    player_gameweek_stats: pd.DataFrame,
    fixture_snapshots: pd.DataFrame | None = None,
    horizons: Sequence[int] = (1, 3, 5),
    recent_window: int = 3,
) -> pd.DataFrame:
    """Build current player features without requiring future labels."""
    if recent_window <= 0:
        raise ValueError("recent_window must be positive")
    normalized_horizons = _validate_horizons(horizons)
    snapshots = player_snapshots.copy()
    stats = player_gameweek_stats.copy()
    fixtures = fixture_snapshots.copy() if fixture_snapshots is not None else None
    _require_columns(
        snapshots,
        {
            "season",
            "snapshot_gameweek",
            "snapshot_timestamp",
            "snapshot_deadline_time",
            "player_id",
        },
        "player snapshots",
    )
    _require_columns(
        stats,
        {"season", "gameweek", "player_id", "total_points", "minutes"},
        "player gameweek stats",
    )
    if snapshots.empty:
        raise FeatureValidationError("Player snapshots contain no rows")
    if fixtures is not None:
        _require_columns(
            snapshots, {"club_id"}, "player snapshots for fixture features"
        )
        _require_columns(
            fixtures,
            {
                "season",
                "snapshot_gameweek",
                "gameweek_id",
                "home_club_id",
                "away_club_id",
            },
            "fixture snapshots",
        )
        for column in (
            "snapshot_gameweek",
            "gameweek_id",
            "home_club_id",
            "away_club_id",
        ):
            fixtures[column] = pd.to_numeric(fixtures[column], errors="coerce")

    snapshots["snapshot_timestamp"] = pd.to_datetime(
        snapshots["snapshot_timestamp"], utc=True, errors="raise"
    )
    snapshots["snapshot_deadline_time"] = pd.to_datetime(
        snapshots["snapshot_deadline_time"], utc=True, errors="raise"
    )
    late = snapshots["snapshot_timestamp"] > snapshots["snapshot_deadline_time"]
    if late.any():
        keys = snapshots.loc[
            late, ["season", "snapshot_gameweek"]
        ].drop_duplicates()
        raise FeatureValidationError(
            "Post-deadline snapshots cannot be used for prediction: "
            + ", ".join(
                f"{row.season} GW{row.snapshot_gameweek}"
                for row in keys.itertuples(index=False)
            )
        )

    snapshot_keys = ["season", "snapshot_gameweek", "player_id"]
    if snapshots.duplicated(snapshot_keys).any():
        raise FeatureValidationError(
            "Player snapshots must be unique by season, gameweek, and player"
        )
    stat_keys = ["season", "gameweek", "player_id"]
    if stats.duplicated(stat_keys).any():
        raise FeatureValidationError(
            "Player gameweek stats must be unique by season, gameweek, and player"
        )
    for column in ("snapshot_gameweek", "player_id"):
        snapshots[column] = pd.to_numeric(snapshots[column], errors="raise")
    for column in ("gameweek", "player_id", "total_points", "minutes"):
        stats[column] = pd.to_numeric(stats[column], errors="raise")
    stats["_appeared"] = stats["minutes"].gt(0).astype(float)
    if "starts" in stats:
        starts = pd.to_numeric(stats["starts"], errors="coerce")
        stats["_started"] = starts.gt(0).astype(float)
    else:
        stats["_started"] = stats["minutes"].ge(60).astype(float)

    derived_groups: list[pd.DataFrame] = []
    for (season, gameweek), snapshot_group in snapshots.groupby(
        ["season", "snapshot_gameweek"], sort=True
    ):
        group = snapshot_group.copy()
        if fixtures is not None:
            group = _add_fixture_features(
                group,
                fixtures,
                str(season),
                int(gameweek),
                normalized_horizons,
            )
        season_stats = stats.loc[stats["season"] == season]
        recent = season_stats.loc[
            (season_stats["gameweek"] < gameweek)
            & (season_stats["gameweek"] >= gameweek - recent_window)
        ]
        recent_features = recent.groupby("player_id", as_index=False).agg(
            **{
                f"recent_points_mean_{recent_window}": (
                    "total_points",
                    "mean",
                ),
                f"recent_minutes_mean_{recent_window}": ("minutes", "mean"),
                f"recent_appearance_rate_{recent_window}": (
                    "_appeared",
                    "mean",
                ),
                f"recent_start_rate_{recent_window}": ("_started", "mean"),
                "recent_gameweeks_available": ("gameweek", "nunique"),
            }
        )
        group = group.merge(
            recent_features, on="player_id", how="left", validate="one_to_one"
        )
        derived_groups.append(group)
    return pd.concat(derived_groups, ignore_index=True).sort_values(
        snapshot_keys, ignore_index=True
    )


def _add_fixture_features(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    season: str,
    snapshot_gameweek: int,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Count known upcoming home/away fixtures for each player's club."""
    result = players.copy()
    snapshot_fixtures = fixtures.loc[
        (fixtures["season"] == season)
        & (fixtures["snapshot_gameweek"] == snapshot_gameweek)
    ]
    for horizon in horizons:
        relevant = snapshot_fixtures.loc[
            (snapshot_fixtures["gameweek_id"] >= snapshot_gameweek)
            & (
                snapshot_fixtures["gameweek_id"]
                < snapshot_gameweek + horizon
            )
        ]
        home_counts = relevant["home_club_id"].value_counts()
        away_counts = relevant["away_club_id"].value_counts()
        home_column = f"upcoming_home_fixture_count_{horizon}"
        away_column = f"upcoming_away_fixture_count_{horizon}"
        total_column = f"upcoming_fixture_count_{horizon}"
        result[home_column] = (
            result["club_id"].map(home_counts).fillna(0).astype("int64")
        )
        result[away_column] = (
            result["club_id"].map(away_counts).fillna(0).astype("int64")
        )
        result[total_column] = result[home_column] + result[away_column]
    return add_team_strength_features(
        result,
        snapshot_fixtures,
        snapshot_gameweek,
        horizons,
    )


def _validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    values = tuple(horizons)
    if not values:
        raise ValueError("At least one horizon is required")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("Horizons must be positive integers")
    if len(set(values)) != len(values):
        raise ValueError("Horizons must not contain duplicates")
    return values


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise FeatureValidationError(
            f"{label.title()} are missing columns: {', '.join(sorted(missing))}"
        )
