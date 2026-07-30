"""Leakage-conscious reconstruction of training rows from historical archives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np
import pandas as pd

from fpl_predictions.data.storage import (
    SnapshotStorageError,
    utc_snapshot_id,
    write_json,
    write_parquet,
)
from fpl_predictions.sources.vaastav import (
    SOURCE_NAME,
    audit_season_directory,
    verify_download,
)

SUM_STAT_COLUMNS = (
    "total_points",
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
)


@dataclass(frozen=True, slots=True)
class BackfillResult:
    """Outputs from one immutable multi-season historical reconstruction."""

    ingestion_id: str
    directory: Path
    training_table: Path
    player_gameweeks: Path
    seasons: tuple[str, ...]


class BackfillValidationError(ValueError):
    """Raised when archived rows cannot be reconstructed safely."""


def build_historical_backfill(
    source_root: Path,
    data_dir: Path,
    seasons: Sequence[str],
    horizons: Sequence[int] = (1, 3, 5),
    recent_window: int = 3,
    verify_source: bool = True,
    created_at: datetime | None = None,
) -> BackfillResult:
    """Audit archived seasons and build past-only features plus future labels."""
    normalized_seasons = tuple(seasons)
    if not normalized_seasons:
        raise ValueError("At least one historical season is required")
    normalized_horizons = _validate_horizons(horizons)
    if recent_window <= 0:
        raise ValueError("recent_window must be positive")
    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    source_manifest = (
        verify_download(source_root, list(normalized_seasons))
        if verify_source
        else None
    )

    audits = []
    season_tables = []
    for season in normalized_seasons:
        season_dir = source_root / season
        audits.append(audit_season_directory(season_dir, season))
        season_tables.append(
            reconstruct_season(
                season_dir,
                season,
                horizons=normalized_horizons,
                recent_window=recent_window,
            )
        )

    training = pd.concat(season_tables, ignore_index=True, sort=False)
    key = ["season", "snapshot_gameweek", "player_id"]
    if training.duplicated(key).any():
        raise BackfillValidationError(
            "Historical reconstruction produced duplicate player snapshot rows"
        )
    training = training.sort_values(key, ignore_index=True)
    player_gameweeks = _player_gameweek_output(training)

    ingestion_id = utc_snapshot_id(timestamp)
    output_dir = (
        data_dir / "processed" / "historical_backfill" / ingestion_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        training_path = output_dir / "training.parquet"
        player_gameweeks_path = output_dir / "player_gameweek_stats.parquet"
        write_parquet(training_path, training)
        write_parquet(player_gameweeks_path, player_gameweeks)
        write_json(output_dir / "audit.json", {"seasons": audits})
        write_json(
            output_dir / "manifest.json",
            {
                "ingestion_id": ingestion_id,
                "created_at_utc": timestamp.astimezone(timezone.utc).isoformat(),
                "source": SOURCE_NAME,
                "source_revision": (
                    source_manifest.get("revision")
                    if source_manifest is not None
                    else None
                ),
                "source_verified": verify_source,
                "seasons": list(normalized_seasons),
                "horizons": list(normalized_horizons),
                "recent_window": recent_window,
                "rows": len(training),
                "players": int(
                    training[["season", "player_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "limitations": [
                    "This is a reconstruction, not an archived pre-deadline snapshot.",
                    "xP is excluded because the source documents timing uncertainty.",
                    "Same-GW performance is never used as a feature.",
                    "Price after GW1 is lagged from the prior archived gameweek.",
                    "Historical fixture counts reflect the final recorded schedule.",
                    "Historical injury news and availability are not available.",
                ],
            },
        )
    except Exception as exc:
        shutil.rmtree(output_dir, ignore_errors=True)
        if isinstance(exc, SnapshotStorageError):
            raise
        raise SnapshotStorageError(
            f"Could not save historical backfill {ingestion_id}: {exc}"
        ) from exc
    return BackfillResult(
        ingestion_id,
        output_dir,
        training_path,
        player_gameweeks_path,
        normalized_seasons,
    )


def reconstruct_season(
    season_dir: Path,
    season: str,
    horizons: Sequence[int] = (1, 3, 5),
    recent_window: int = 3,
) -> pd.DataFrame:
    """Reconstruct prediction rows from match-level archived gameweek data."""
    normalized_horizons = _validate_horizons(horizons)
    audit_season_directory(season_dir, season)
    fixtures = pd.read_csv(season_dir / "fixtures.csv", low_memory=False)
    raw = pd.read_csv(
        season_dir / "gws" / "merged_gw.csv",
        low_memory=False,
    )
    raw = raw.drop(columns=["xP"], errors="ignore")
    for column in ("element", "round", "fixture", "value"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if raw[["element", "round"]].isna().any().any():
        raise BackfillValidationError(
            f"{season} contains invalid player or gameweek IDs"
        )
    raw["kickoff_time"] = pd.to_datetime(
        raw["kickoff_time"], utc=True, errors="coerce"
    )
    period_times = raw.groupby("round")["kickoff_time"].min()
    if period_times.isna().any():
        missing_rounds = period_times[period_times.isna()].index.tolist()
        raise BackfillValidationError(
            f"{season} has no valid kickoff time for gameweeks: {missing_rounds}"
        )
    raw["kickoff_time"] = raw["kickoff_time"].fillna(
        raw["round"].map(period_times)
    )
    known_rounds = set(raw["round"].dropna().astype(int))
    fixture_rounds = set(
        pd.to_numeric(fixtures["event"], errors="coerce")
        .dropna()
        .astype(int)
    )
    league_wide_blanks = set(
        range(int(raw["round"].min()), int(raw["round"].max()) + 1)
    ).difference(known_rounds | fixture_rounds)
    available_stats = [
        column for column in SUM_STAT_COLUMNS if column in raw.columns
    ]
    for column in available_stats:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["was_home"] = _boolean_series(raw["was_home"])

    group_key = ["element", "round"]
    consistency = raw.groupby(group_key).agg(
        names=("name", "nunique"),
        positions=("position", "nunique"),
        teams=("team", "nunique"),
    )
    if (consistency > 1).any(axis=None):
        raise BackfillValidationError(
            f"{season} has inconsistent identity values within a player gameweek"
        )

    aggregation: dict[str, tuple[str, Any]] = {
        "display_name": ("name", "first"),
        "position_short_name": ("position", "first"),
        "club_name": ("team", "first"),
        "archived_price_tenths": ("value", "first"),
        "first_kickoff": ("kickoff_time", "min"),
        "fixture_count": ("fixture", "nunique"),
        "home_fixture_count": ("was_home", "sum"),
    }
    for column in available_stats:
        aggregation[f"gameweek_{column}"] = (
            column,
            lambda values: values.sum(min_count=1),
        )
    gameweeks = (
        raw.groupby(group_key, as_index=False)
        .agg(**aggregation)
        .rename(columns={"element": "player_id", "round": "snapshot_gameweek"})
    )
    gameweeks["away_fixture_count"] = (
        gameweeks["fixture_count"] - gameweeks["home_fixture_count"]
    )
    gameweeks["season"] = season

    period_times = period_times.to_dict()
    gameweeks["snapshot_timestamp"] = gameweeks["snapshot_gameweek"].map(
        period_times
    )
    gameweeks["historical_time_proxy"] = "first_fixture_kickoff"
    gameweeks["historical_reconstruction"] = True
    gameweeks["data_source"] = SOURCE_NAME

    gameweeks = gameweeks.sort_values(
        ["player_id", "snapshot_gameweek"], ignore_index=True
    )
    grouped = gameweeks.groupby("player_id", sort=False)
    for stat in available_stats:
        event_column = f"gameweek_{stat}"
        gameweeks[stat] = grouped[event_column].transform(
            lambda values: values.fillna(0).cumsum().shift(1)
        )
        gameweeks[stat] = gameweeks[stat].fillna(0)

    gameweeks["price"] = grouped["archived_price_tenths"].shift(1) / 10.0
    first_player_row = grouped.cumcount() == 0
    gameweeks.loc[first_player_row, "price"] = (
        gameweeks.loc[first_player_row, "archived_price_tenths"] / 10.0
    )
    points_column = "gameweek_total_points"
    minutes_column = "gameweek_minutes"
    recent_points, recent_available = _past_window_mean(
        gameweeks,
        points_column,
        recent_window,
        league_wide_blanks,
    )
    recent_minutes, _ = _past_window_mean(
        gameweeks,
        minutes_column,
        recent_window,
        league_wide_blanks,
    )
    gameweeks[f"recent_points_mean_{recent_window}"] = recent_points
    gameweeks[f"recent_minutes_mean_{recent_window}"] = recent_minutes
    gameweeks["recent_gameweeks_available"] = recent_available

    for horizon in normalized_horizons:
        gameweeks[f"upcoming_fixture_count_{horizon}"] = _future_window_sum(
            gameweeks,
            "fixture_count",
            horizon,
            league_wide_blanks,
        )
        gameweeks[f"upcoming_home_fixture_count_{horizon}"] = (
            _future_window_sum(
                gameweeks,
                "home_fixture_count",
                horizon,
                league_wide_blanks,
            )
        )
        gameweeks[f"upcoming_away_fixture_count_{horizon}"] = (
            _future_window_sum(
                gameweeks,
                "away_fixture_count",
                horizon,
                league_wide_blanks,
            )
        )
        label = _label_column(horizon)
        gameweeks[label] = _future_window_sum(
            gameweeks,
            points_column,
            horizon,
            league_wide_blanks,
        )

    return gameweeks


def _future_window_sum(
    frame: pd.DataFrame,
    value_column: str,
    horizon: int,
    zero_gameweeks: set[int],
) -> pd.Series:
    values = [
        _offset_values(
            frame,
            value_column,
            offset=-offset,
            zero_gameweeks=zero_gameweeks,
        )
        for offset in range(horizon)
    ]
    window = pd.concat(values, axis=1)
    return window.sum(axis=1, min_count=horizon)


def _past_window_mean(
    frame: pd.DataFrame,
    value_column: str,
    window: int,
    zero_gameweeks: set[int],
) -> tuple[pd.Series, pd.Series]:
    values = [
        _offset_values(
            frame,
            value_column,
            offset=offset,
            zero_gameweeks=zero_gameweeks,
        )
        for offset in range(1, window + 1)
    ]
    history = pd.concat(values, axis=1)
    return history.mean(axis=1), history.count(axis=1).astype(float)


def _offset_values(
    frame: pd.DataFrame,
    value_column: str,
    offset: int,
    zero_gameweeks: set[int],
) -> pd.Series:
    """Look up values at target gameweek minus offset without filling unknowns."""
    base = frame[["player_id", "snapshot_gameweek"]].copy()
    base["_row_index"] = frame.index
    lookup = frame[
        ["player_id", "snapshot_gameweek", value_column]
    ].copy()
    lookup["snapshot_gameweek"] = lookup["snapshot_gameweek"] + offset
    lookup = lookup.rename(columns={value_column: "_offset_value"})
    merged = base.merge(
        lookup,
        on=["player_id", "snapshot_gameweek"],
        how="left",
        validate="one_to_one",
    ).set_index("_row_index")
    result = pd.to_numeric(
        merged["_offset_value"], errors="coerce"
    ).reindex(frame.index)
    source_gameweek = frame["snapshot_gameweek"] - offset
    explicit_blank = source_gameweek.isin(zero_gameweeks)
    return result.mask(explicit_blank & result.isna(), 0.0)


def _player_gameweek_output(training: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "season",
        "snapshot_gameweek",
        "player_id",
        "display_name",
        "position_short_name",
        "club_name",
    ]
    event_columns = [
        column for column in training if column.startswith("gameweek_")
    ]
    result = training.loc[:, columns + event_columns].rename(
        columns={"snapshot_gameweek": "gameweek"}
    )
    return result


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    invalid = ~normalized.isin(["true", "false"]) & normalized.notna()
    if invalid.any():
        raise BackfillValidationError(
            "Historical was_home contains invalid boolean values"
        )
    return normalized.eq("true").fillna(False)


def _validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    values = tuple(horizons)
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("horizons must contain positive integers")
    if len(values) != len(set(values)):
        raise ValueError("horizons must not contain duplicates")
    return values


def _label_column(horizon: int) -> str:
    suffix = "gameweek" if horizon == 1 else "gameweeks"
    return f"label_next_{horizon}_{suffix}"
