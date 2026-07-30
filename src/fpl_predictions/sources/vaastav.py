"""Reproducible downloader and schema audit for the Vaastav FPL archive."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fpl_predictions.data.storage import write_json

SOURCE_NAME = "vaastav/Fantasy-Premier-League"
SOURCE_REPOSITORY = "https://github.com/vaastav/Fantasy-Premier-League"
RAW_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/vaastav/"
    "Fantasy-Premier-League/{revision}/data/{season}/{relative_path}"
)
SEASON_PATTERN = re.compile(r"^\d{4}-\d{2}$")
REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
REQUIRED_FILES = ("gws/merged_gw.csv", "fixtures.csv", "teams.csv")
REQUIRED_GAMEWEEK_COLUMNS = {
    "name",
    "position",
    "team",
    "element",
    "fixture",
    "round",
    "kickoff_time",
    "minutes",
    "total_points",
    "value",
    "was_home",
}
REQUIRED_FIXTURE_COLUMNS = {
    "id",
    "event",
    "team_h",
    "team_a",
    "kickoff_time",
}
REQUIRED_TEAM_COLUMNS = {"id", "name", "short_name"}


class HistoricalSourceError(RuntimeError):
    """Raised when an external historical source is unavailable or invalid."""


@dataclass(frozen=True, slots=True)
class DownloadedSeason:
    """Files and provenance for one downloaded historical season."""

    season: str
    directory: Path
    files: tuple[dict[str, Any], ...]


def download_seasons(
    destination: Path,
    seasons: list[str],
    revision: str,
    timeout_seconds: float = 60.0,
    session: requests.Session | None = None,
) -> Path:
    """Download only required archive files at an immutable Git revision."""
    _validate_revision(revision)
    if not seasons:
        raise ValueError("At least one historical season is required")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    normalized_seasons = [_validate_season(season) for season in seasons]
    source_root = destination / SOURCE_NAME.replace("/", "__") / revision
    if source_root.exists():
        raise FileExistsError(
            f"Historical download already exists: {source_root}"
        )
    source_root.mkdir(parents=True)
    http = session or _session()
    downloaded: list[DownloadedSeason] = []
    try:
        for season in normalized_seasons:
            season_dir = source_root / season
            season_dir.mkdir()
            file_records = []
            for relative_path in REQUIRED_FILES:
                url = RAW_URL_TEMPLATE.format(
                    revision=revision,
                    season=season,
                    relative_path=relative_path,
                )
                target = season_dir / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                content = _download_bytes(http, url, timeout_seconds)
                _atomic_bytes_write(target, content)
                file_records.append(
                    {
                        "relative_path": relative_path,
                        "url": url,
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
            downloaded.append(
                DownloadedSeason(season, season_dir, tuple(file_records))
            )
        write_json(
            source_root / "source_manifest.json",
            {
                "source": SOURCE_NAME,
                "repository": SOURCE_REPOSITORY,
                "revision": revision.lower(),
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "license": f"{SOURCE_REPOSITORY}/blob/{revision}/LICENSE",
                "seasons": [
                    {"season": item.season, "files": list(item.files)}
                    for item in downloaded
                ],
            },
        )
    except Exception:
        shutil.rmtree(source_root, ignore_errors=True)
        raise
    return source_root


def verify_download(
    source_root: Path,
    seasons: list[str],
) -> dict[str, Any]:
    """Verify source identity and every requested file against its manifest."""
    manifest_path = source_root / "source_manifest.json"
    if not manifest_path.is_file():
        raise HistoricalSourceError(
            f"Historical source manifest not found: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source") != SOURCE_NAME:
        raise HistoricalSourceError(
            f"Unexpected historical source: {manifest.get('source')!r}"
        )
    _validate_revision(str(manifest.get("revision", "")))
    season_entries = {
        entry.get("season"): entry
        for entry in manifest.get("seasons", [])
        if isinstance(entry, dict)
    }
    for season in seasons:
        normalized = _validate_season(season)
        entry = season_entries.get(normalized)
        if entry is None:
            raise HistoricalSourceError(
                f"Source manifest does not include season {normalized}"
            )
        for file_record in entry.get("files", []):
            relative_path = file_record.get("relative_path")
            expected_checksum = file_record.get("sha256")
            if relative_path not in REQUIRED_FILES:
                continue
            path = source_root / normalized / relative_path
            if not path.is_file():
                raise HistoricalSourceError(f"Downloaded file is missing: {path}")
            actual_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_checksum != expected_checksum:
                raise HistoricalSourceError(
                    f"Checksum mismatch for {normalized}/{relative_path}"
                )
    return manifest


def audit_season_directory(season_dir: Path, season: str) -> dict[str, Any]:
    """Validate required files and report feature coverage without assumptions."""
    normalized_season = _validate_season(season)
    paths = {
        relative: season_dir / relative for relative in REQUIRED_FILES
    }
    missing_files = [
        relative for relative, path in paths.items() if not path.is_file()
    ]
    if missing_files:
        raise HistoricalSourceError(
            f"{normalized_season} is missing source files: "
            + ", ".join(missing_files)
        )

    gameweeks = pd.read_csv(paths["gws/merged_gw.csv"], low_memory=False)
    fixtures = pd.read_csv(paths["fixtures.csv"], low_memory=False)
    teams = pd.read_csv(paths["teams.csv"], low_memory=False)
    _require_columns(
        gameweeks, REQUIRED_GAMEWEEK_COLUMNS, "merged gameweek"
    )
    _require_columns(fixtures, REQUIRED_FIXTURE_COLUMNS, "fixtures")
    _require_columns(teams, REQUIRED_TEAM_COLUMNS, "teams")
    if gameweeks.empty:
        raise HistoricalSourceError(
            f"{normalized_season} merged gameweek file is empty"
        )

    duplicate_key = ["element", "round", "fixture"]
    duplicates = int(gameweeks.duplicated(duplicate_key).sum())
    feature_candidates = [
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "ict_index",
        "influence",
        "creativity",
        "threat",
        "starts",
        "selected",
        "transfers_in",
        "transfers_out",
    ]
    coverage = {
        column: (
            float(gameweeks[column].notna().mean())
            if column in gameweeks
            else None
        )
        for column in feature_candidates
    }
    rounds = pd.to_numeric(gameweeks["round"], errors="coerce")
    fixture_events = set(
        pd.to_numeric(fixtures["event"], errors="coerce")
        .dropna()
        .astype(int)
    )
    first_gameweek = int(rounds.min())
    last_gameweek = int(rounds.max())
    missing_gameweeks = sorted(
        set(range(first_gameweek, last_gameweek + 1))
        .difference(rounds.dropna().astype(int))
    )
    league_wide_blanks = [
        gameweek
        for gameweek in missing_gameweeks
        if gameweek not in fixture_events
    ]
    return {
        "season": normalized_season,
        "rows": len(gameweeks),
        "players": int(gameweeks["element"].nunique()),
        "gameweeks": int(rounds.nunique()),
        "first_gameweek": first_gameweek,
        "last_gameweek": last_gameweek,
        "missing_gameweek_files": missing_gameweeks,
        "league_wide_blank_gameweeks": league_wide_blanks,
        "fixtures": int(fixtures["id"].nunique()),
        "teams": int(teams["id"].nunique()),
        "duplicate_player_gameweek_fixture_rows": duplicates,
        "column_count": len(gameweeks.columns),
        "feature_coverage": coverage,
        "excluded_columns": {
            "xP": (
                "Potential post-match information documented by the source; "
                "never used unshifted."
            )
        },
    }


def _download_bytes(
    session: requests.Session,
    url: str,
    timeout_seconds: float,
) -> bytes:
    try:
        response = session.get(url, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HistoricalSourceError(
            f"Historical source request failed for {url}: {exc}"
        ) from exc
    if not response.content:
        raise HistoricalSourceError(
            f"Historical source returned an empty file: {url}"
        )
    return response.content


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "fpl-predictions-history/0.1"})
    return session


def _atomic_bytes_write(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    temporary.replace(path)


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise HistoricalSourceError(
            f"Vaastav {label} data is missing columns: "
            + ", ".join(sorted(missing))
        )


def _validate_season(season: str) -> str:
    if not SEASON_PATTERN.fullmatch(season):
        raise ValueError("season must use YYYY-YY format, for example 2023-24")
    start = int(season[:4])
    expected_end = (start + 1) % 100
    if int(season[-2:]) != expected_end:
        raise ValueError(f"season has inconsistent years: {season}")
    return season


def _validate_revision(revision: str) -> None:
    if not REVISION_PATTERN.fullmatch(revision):
        raise ValueError(
            "revision must be a complete 40-character Git commit SHA"
        )
