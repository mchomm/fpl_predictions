"""Ingestion and normalization of finalized official FPL gameweek data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
from typing import Any

import pandas as pd

from fpl_predictions.api.client import FPLClient
from fpl_predictions.data.normalize import (
    NormalizationError,
    normalize_clubs,
    normalize_fixtures,
    normalize_gameweeks,
)
from fpl_predictions.data.storage import (
    SnapshotStorageError,
    utc_snapshot_id,
    write_json,
    write_parquet,
)

SEASON_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class HistoryIngestion:
    """Location and coverage of one immutable historical ingestion run."""

    ingestion_id: str
    season: str
    gameweeks: tuple[int, ...]
    raw_dir: Path
    processed_dir: Path


def normalize_live_gameweek(
    payload: dict[str, Any],
    season: str,
    gameweek: int,
    ingested_at: datetime,
) -> pd.DataFrame:
    """Flatten aggregate player stats from an official gameweek live response."""
    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise NormalizationError("Live gameweek response elements must be a list")

    rows: list[dict[str, Any]] = []
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            raise NormalizationError(f"Live element {index} must be an object")
        player_id = element.get("id")
        stats = element.get("stats")
        if isinstance(player_id, bool) or not isinstance(player_id, int):
            raise NormalizationError(f"Live element {index} has an invalid player ID")
        if not isinstance(stats, dict):
            raise NormalizationError(
                f"Live element for player {player_id} has no stats object"
            )
        rows.append(
            {
                "season": season,
                "gameweek": gameweek,
                "player_id": player_id,
                "ingested_at": ingested_at.astimezone(timezone.utc).isoformat(),
                **stats,
                "explain": element.get("explain"),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise NormalizationError(f"Gameweek {gameweek} contains no player stats")
    if result["player_id"].duplicated().any():
        duplicate_ids = sorted(
            result.loc[result["player_id"].duplicated(), "player_id"].unique()
        )
        raise NormalizationError(
            f"Gameweek {gameweek} contains duplicate player IDs: {duplicate_ids}"
        )
    return result.sort_values("player_id", ignore_index=True)


def ingest_finished_gameweeks(
    client: FPLClient,
    data_dir: Path,
    season: str,
    through_gameweek: int | None = None,
    ingested_at: datetime | None = None,
) -> HistoryIngestion:
    """Ingest finalized live stats and fixture metadata into an immutable run."""
    validate_season(season)
    if through_gameweek is not None and through_gameweek <= 0:
        raise ValueError("through_gameweek must be positive")
    timestamp = ingested_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("ingested_at must be timezone-aware")

    bootstrap = client.get_bootstrap()
    fixtures_payload = client.get_fixtures()
    events = bootstrap["events"]
    finished = sorted(
        int(event["id"])
        for event in events
        if event.get("finished") is True
        and (through_gameweek is None or int(event["id"]) <= through_gameweek)
    )
    if not finished:
        raise NormalizationError("No finished gameweeks matched the requested range")
    if through_gameweek is not None and through_gameweek not in finished:
        raise NormalizationError(
            f"Gameweek {through_gameweek} is not marked finished by the API"
        )

    ingestion_id = utc_snapshot_id(timestamp)
    raw_dir = data_dir / "raw" / "history" / season / ingestion_id
    processed_dir = data_dir / "processed" / "history" / season / ingestion_id
    raw_dir.mkdir(parents=True, exist_ok=False)
    processed_dir.mkdir(parents=True, exist_ok=False)

    try:
        write_json(raw_dir / "bootstrap-static.json", bootstrap)
        write_json(raw_dir / "fixtures.json", fixtures_payload)

        clubs = normalize_clubs(bootstrap["teams"])
        gameweeks = normalize_gameweeks(events)
        fixtures = normalize_fixtures(fixtures_payload, clubs)
        for table in (clubs, gameweeks, fixtures):
            table.insert(0, "ingested_at", timestamp.isoformat())
            table.insert(0, "season", season)
        write_parquet(processed_dir / "clubs.parquet", clubs)
        write_parquet(processed_dir / "gameweeks.parquet", gameweeks)
        write_parquet(processed_dir / "fixtures.parquet", fixtures)

        gameweek_tables = []
        live_dir = raw_dir / "live"
        live_dir.mkdir()
        for gameweek in finished:
            live_payload = client.get_live_gameweek(gameweek)
            write_json(live_dir / f"gameweek-{gameweek:02d}.json", live_payload)
            gameweek_tables.append(
                normalize_live_gameweek(
                    live_payload, season, gameweek, timestamp
                )
            )
        player_stats = pd.concat(gameweek_tables, ignore_index=True)
        write_parquet(
            processed_dir / "player_gameweek_stats.parquet", player_stats
        )

        manifest = {
            "ingestion_id": ingestion_id,
            "season": season,
            "ingested_at": timestamp.astimezone(timezone.utc).isoformat(),
            "finished_gameweeks": finished,
            "tables": {
                "clubs": len(clubs),
                "gameweeks": len(gameweeks),
                "fixtures": len(fixtures),
                "player_gameweek_stats": len(player_stats),
            },
        }
        write_json(processed_dir / "manifest.json", manifest)
    except Exception as exc:
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(processed_dir, ignore_errors=True)
        if isinstance(exc, (NormalizationError, SnapshotStorageError)):
            raise
        raise SnapshotStorageError(
            f"Could not ingest history run {ingestion_id}: {exc}"
        ) from exc

    return HistoryIngestion(
        ingestion_id,
        season,
        tuple(finished),
        raw_dir,
        processed_dir,
    )


def validate_season(season: str) -> None:
    if not SEASON_PATTERN.fullmatch(season) or season in {".", ".."}:
        raise ValueError(
            "season must contain only letters, digits, dots, underscores, or hyphens"
        )
