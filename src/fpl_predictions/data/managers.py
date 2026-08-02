"""Privacy-minimized snapshots of public post-deadline manager squads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from fpl_predictions.api.client import FPLClient
from fpl_predictions.data.history import validate_season
from fpl_predictions.data.normalize import NormalizationError
from fpl_predictions.data.storage import (
    SnapshotStorageError,
    utc_snapshot_id,
    write_json,
    write_parquet,
)


@dataclass(frozen=True, slots=True)
class ManagerSampleIngestion:
    """Location and coverage of one immutable manager-squad sample."""

    ingestion_id: str
    season: str
    gameweek: int
    managers: int
    raw_dir: Path
    processed_dir: Path


def discover_overall_league_id(
    client: FPLClient,
    discovery_manager_id: int,
) -> int:
    """Discover the season-specific Overall league without hard-coding its ID."""
    payload = client.get_manager(discovery_manager_id)
    classic = payload["leagues"]["classic"]
    matches = [
        row
        for row in classic
        if isinstance(row, dict)
        and (row.get("short_name") == "overall" or row.get("name") == "Overall")
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("id"), int):
        raise NormalizationError(
            "Could not discover exactly one Overall league from manager metadata"
        )
    return int(matches[0]["id"])


def normalize_manager_picks(
    payload: dict[str, Any],
    manager_id: int,
    season: str,
    gameweek: int,
    collected_at: datetime,
    valid_player_ids: set[int],
    sampling_metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize one public picks response and validate its squad structure."""
    picks = payload.get("picks")
    history = payload.get("entry_history")
    if not isinstance(picks, list) or not isinstance(history, dict):
        raise NormalizationError(
            f"Manager {manager_id} picks response is missing picks or history"
        )
    if len(picks) != 15:
        raise NormalizationError(
            f"Manager {manager_id} must have 15 picks, got {len(picks)}"
        )
    frame = pd.DataFrame(picks).rename(
        columns={"element": "player_id", "position": "squad_position"}
    )
    required = {
        "player_id",
        "squad_position",
        "multiplier",
        "is_captain",
        "is_vice_captain",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise NormalizationError(
            f"Manager {manager_id} picks are missing fields: "
            + ", ".join(sorted(missing))
        )
    for column in ("player_id", "squad_position", "multiplier"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    if frame["player_id"].duplicated().any():
        raise NormalizationError(f"Manager {manager_id} has duplicate players")
    stale = sorted(set(frame["player_id"]).difference(valid_player_ids))
    if stale:
        raise NormalizationError(
            f"Manager {manager_id} has unknown player IDs: {stale}"
        )
    if sorted(frame["squad_position"].tolist()) != list(range(1, 16)):
        raise NormalizationError(
            f"Manager {manager_id} squad positions must be exactly 1-15"
        )
    for column, label in (
        ("is_captain", "captain"),
        ("is_vice_captain", "vice-captain"),
    ):
        values = frame[column].map(bool)
        if int(values.sum()) != 1:
            raise NormalizationError(
                f"Manager {manager_id} must have exactly one {label}"
            )
        frame[column] = values
    timestamp = collected_at.astimezone(timezone.utc).isoformat()
    frame.insert(0, "collected_at", timestamp)
    frame.insert(0, "gameweek", gameweek)
    frame.insert(0, "season", season)
    frame.insert(0, "manager_id", manager_id)
    frame["lineup_role"] = np.where(
        frame["squad_position"] <= 11, "starter", "bench"
    )
    frame = frame.sort_values("squad_position", ignore_index=True)

    manager = {
        "manager_id": manager_id,
        "season": season,
        "gameweek": gameweek,
        "collected_at": timestamp,
        "active_chip": payload.get("active_chip"),
        **(sampling_metadata or {}),
    }
    for key, value in history.items():
        if key not in manager:
            manager[key] = value
    return frame, manager


def collect_manager_sample(
    client: FPLClient,
    data_dir: Path,
    season: str,
    gameweek: int,
    sample_size: int = 250,
    random_seed: int = 2026,
    discovery_manager_id: int = 1,
    manager_ids: Iterable[int] | None = None,
    collected_at: datetime | None = None,
    request_interval_seconds: float = 0.1,
    sleeper: Callable[[float], None] = time.sleep,
) -> ManagerSampleIngestion:
    """Collect a reproducible random Overall sample after the FPL deadline."""
    validate_season(season)
    if gameweek <= 0:
        raise ValueError("gameweek must be positive")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if request_interval_seconds < 0:
        raise ValueError("request_interval_seconds must be non-negative")
    timestamp = collected_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("collected_at must be timezone-aware")

    bootstrap = client.get_bootstrap()
    event = next(
        (row for row in bootstrap["events"] if int(row.get("id", -1)) == gameweek),
        None,
    )
    if not isinstance(event, dict) or not isinstance(event.get("deadline_time"), str):
        raise NormalizationError(f"Gameweek {gameweek} has no API deadline")
    deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
    if timestamp.astimezone(timezone.utc) < deadline.astimezone(timezone.utc):
        raise NormalizationError(
            f"Manager picks cannot be collected before the GW{gameweek} deadline "
            f"at {deadline.isoformat()}"
        )
    valid_player_ids = {
        int(row["id"])
        for row in bootstrap["elements"]
        if isinstance(row, dict) and isinstance(row.get("id"), int)
    }

    supplied_ids = tuple(dict.fromkeys(int(value) for value in manager_ids or ()))
    if supplied_ids:
        if any(value <= 0 for value in supplied_ids):
            raise ValueError("manager IDs must be positive")
        selected = [
            {"manager_id": value, "sample_method": "explicit"}
            for value in supplied_ids
        ]
        overall_league_id = None
        sampled_pages: list[int] = []
        total_players = bootstrap.get("total_players")
    else:
        overall_league_id = discover_overall_league_id(
            client, discovery_manager_id
        )
        selected, sampled_pages, total_players = _sample_overall_standings(
            client,
            overall_league_id,
            bootstrap.get("total_players"),
            sample_size,
            random_seed,
        )

    ingestion_id = utc_snapshot_id(timestamp)
    raw_dir = (
        data_dir
        / "raw"
        / "manager_samples"
        / season
        / f"gw-{gameweek:02d}"
        / ingestion_id
    )
    processed_dir = (
        data_dir
        / "processed"
        / "manager_samples"
        / season
        / f"gw-{gameweek:02d}"
        / ingestion_id
    )
    raw_dir.mkdir(parents=True, exist_ok=False)
    processed_dir.mkdir(parents=True, exist_ok=False)
    try:
        write_json(raw_dir / "bootstrap-static.json", bootstrap)
        write_json(raw_dir / "sampling-frame.json", selected)
        picks_dir = raw_dir / "picks"
        picks_dir.mkdir()
        pick_tables = []
        manager_rows = []
        for index, sample in enumerate(selected):
            manager_id = int(sample["manager_id"])
            if index and request_interval_seconds:
                sleeper(request_interval_seconds)
            payload = client.get_manager_picks(manager_id, gameweek)
            write_json(picks_dir / f"manager-{manager_id}.json", payload)
            picks, manager = normalize_manager_picks(
                payload,
                manager_id,
                season,
                gameweek,
                timestamp,
                valid_player_ids,
                sample,
            )
            pick_tables.append(picks)
            manager_rows.append(manager)
        all_picks = pd.concat(pick_tables, ignore_index=True)
        managers = pd.DataFrame(manager_rows).sort_values(
            "manager_id", ignore_index=True
        )
        all_picks, managers, player_rates, audit = _enrich_manager_sample(
            all_picks,
            managers,
            bootstrap,
        )
        write_parquet(processed_dir / "manager_picks.parquet", all_picks)
        write_parquet(processed_dir / "managers.parquet", managers)
        write_parquet(processed_dir / "sample_player_rates.parquet", player_rates)
        write_json(processed_dir / "audit.json", audit)
        manifest = {
            "ingestion_id": ingestion_id,
            "season": season,
            "gameweek": gameweek,
            "collected_at": timestamp.astimezone(timezone.utc).isoformat(),
            "deadline_time": deadline.astimezone(timezone.utc).isoformat(),
            "sample_method": "explicit" if supplied_ids else "uniform_overall_pages",
            "sample_size": len(managers),
            "random_seed": random_seed,
            "discovery_manager_id": (
                None if supplied_ids else discovery_manager_id
            ),
            "overall_league_id": overall_league_id,
            "sampled_pages": sampled_pages,
            "api_total_players": total_players,
            "privacy": (
                "Manager and team names are intentionally excluded; only public "
                "entry IDs, ranks, gameweek history, and picks are stored."
            ),
            "tables": {
                "managers": len(managers),
                "manager_picks": len(all_picks),
                "sample_player_rates": len(player_rates),
            },
        }
        write_json(processed_dir / "manifest.json", manifest)
    except Exception as exc:
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(processed_dir, ignore_errors=True)
        if isinstance(exc, (NormalizationError, SnapshotStorageError)):
            raise
        raise SnapshotStorageError(
            f"Could not collect manager sample {ingestion_id}: {exc}"
        ) from exc
    return ManagerSampleIngestion(
        ingestion_id,
        season,
        gameweek,
        len(managers),
        raw_dir,
        processed_dir,
    )


def _enrich_manager_sample(
    picks: pd.DataFrame,
    managers: pd.DataFrame,
    bootstrap: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Add snapshot-time player context and audit empirical selection rates."""
    elements = pd.DataFrame(bootstrap["elements"])
    required = {
        "id",
        "team",
        "element_type",
        "now_cost",
        "selected_by_percent",
    }
    missing = required.difference(elements.columns)
    if missing:
        raise NormalizationError(
            "bootstrap elements are missing manager-audit fields: "
            + ", ".join(sorted(missing))
        )
    metadata = elements.loc[:, sorted(required)].rename(
        columns={
            "id": "player_id",
            "team": "club_id",
            "element_type": "position_id",
            "now_cost": "current_price",
            "selected_by_percent": "api_ownership_percent",
        }
    )
    metadata["current_price"] = pd.to_numeric(
        metadata["current_price"], errors="raise"
    ) / 10.0
    metadata["api_ownership_percent"] = pd.to_numeric(
        metadata["api_ownership_percent"], errors="coerce"
    )
    enriched = picks.merge(metadata, on="player_id", how="left", validate="many_to_one")
    if enriched["position_id"].isna().any():
        raise NormalizationError("Could not enrich every sampled manager pick")

    positions = {
        int(row["id"]): str(row["singular_name_short"])
        for row in bootstrap["element_types"]
        if isinstance(row, dict)
        and isinstance(row.get("id"), int)
        and row.get("singular_name_short") is not None
    }
    enriched["position_short_name"] = enriched["position_id"].map(positions)
    if enriched["position_short_name"].isna().any():
        raise NormalizationError("Could not map sampled pick positions")

    squad_costs = enriched.groupby("manager_id")["current_price"].sum()
    managers["current_squad_cost"] = managers["manager_id"].map(squad_costs)
    starter_counts = (
        enriched[enriched["lineup_role"] == "starter"]
        .groupby(["manager_id", "position_short_name"])
        .size()
        .unstack(fill_value=0)
    )
    outfield_names = [
        row["singular_name_short"]
        for row in bootstrap["element_types"]
        if isinstance(row, dict) and row.get("singular_name_short") != "GKP"
    ]
    formations = starter_counts.reindex(
        columns=outfield_names, fill_value=0
    ).astype(int).astype(str).agg("-".join, axis=1)
    managers["formation"] = managers["manager_id"].map(formations)

    manager_count = len(managers)
    grouped = enriched.groupby("player_id")
    player_rates = grouped.agg(
        selection_count=("manager_id", "nunique"),
        starter_count=("lineup_role", lambda value: int((value == "starter").sum())),
        captain_count=("is_captain", "sum"),
        vice_captain_count=("is_vice_captain", "sum"),
        club_id=("club_id", "first"),
        position_id=("position_id", "first"),
        position_short_name=("position_short_name", "first"),
        current_price=("current_price", "first"),
        api_ownership_percent=("api_ownership_percent", "first"),
    ).reset_index()
    player_rates.insert(0, "collected_at", enriched["collected_at"].iloc[0])
    player_rates.insert(0, "gameweek", int(enriched["gameweek"].iloc[0]))
    player_rates.insert(0, "season", str(enriched["season"].iloc[0]))
    for count_column, rate_column in (
        ("selection_count", "sample_ownership_percent"),
        ("starter_count", "sample_starter_percent"),
        ("captain_count", "sample_captain_percent"),
        ("vice_captain_count", "sample_vice_captain_percent"),
    ):
        player_rates[rate_column] = (
            100.0 * player_rates[count_column] / manager_count
        )
    calibration = player_rates.dropna(subset=["api_ownership_percent"])
    correlation = calibration[
        ["api_ownership_percent", "sample_ownership_percent"]
    ].corr().iloc[0, 1]
    audit = {
        "managers": manager_count,
        "picks": len(enriched),
        "unique_players_selected": int(enriched["player_id"].nunique()),
        "formations": {
            str(key): int(value)
            for key, value in managers["formation"].value_counts().items()
        },
        "current_squad_cost": {
            "minimum": float(managers["current_squad_cost"].min()),
            "median": float(managers["current_squad_cost"].median()),
            "maximum": float(managers["current_squad_cost"].max()),
        },
        "ownership_calibration": {
            "pearson_correlation": (
                float(correlation) if pd.notna(correlation) else None
            ),
            "mean_absolute_error_percentage_points": float(
                (
                    calibration["api_ownership_percent"]
                    - calibration["sample_ownership_percent"]
                ).abs().mean()
            ),
        },
    }
    return enriched, managers, player_rates, audit


def _sample_overall_standings(
    client: FPLClient,
    league_id: int,
    total_players_value: Any,
    sample_size: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[int], int]:
    """Sample standings pages uniformly, then entries uniformly within them."""
    if not isinstance(total_players_value, int) or total_players_value <= 0:
        raise NormalizationError("bootstrap total_players must be positive")
    first = client.get_classic_league_standings(league_id, 1)
    first_results = first["standings"]["results"]
    if not first_results:
        raise NormalizationError(
            "Overall standings are empty; collect only after public ranks exist"
        )
    page_size = len(first_results)
    maximum_page = math.ceil(total_players_value / page_size)
    pages_needed = math.ceil(sample_size / page_size)
    if pages_needed > maximum_page:
        raise NormalizationError("sample_size exceeds the Overall population")
    rng = np.random.default_rng(random_seed)
    pages = sorted(
        int(value)
        for value in rng.choice(
            np.arange(1, maximum_page + 1),
            size=pages_needed,
            replace=False,
        )
    )
    rows = []
    for page in pages:
        payload = first if page == 1 else client.get_classic_league_standings(
            league_id, page
        )
        for row in payload["standings"]["results"]:
            manager_id = row.get("entry")
            if isinstance(manager_id, int):
                rows.append(
                    {
                        "manager_id": manager_id,
                        "sample_method": "uniform_overall_pages",
                        "sample_page": page,
                        "sample_rank": row.get("rank"),
                        "sample_total_points": row.get("total"),
                        "sample_event_points": row.get("event_total"),
                    }
                )
    unique = {row["manager_id"]: row for row in rows}
    candidates = list(unique.values())
    if len(candidates) < sample_size:
        raise NormalizationError(
            f"Sampled pages yielded only {len(candidates)} unique managers"
        )
    chosen = rng.choice(len(candidates), size=sample_size, replace=False)
    return [candidates[int(index)] for index in chosen], pages, total_players_value
