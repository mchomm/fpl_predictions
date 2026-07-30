"""Deadline-aware gameweek snapshot metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd


class SnapshotValidationError(ValueError):
    """Raised when a gameweek snapshot would not represent prediction-time data."""


@dataclass(frozen=True, slots=True)
class GameweekSnapshot:
    """Identity and timing for one pre-deadline prediction snapshot."""

    season: str
    snapshot_gameweek: int
    snapshot_timestamp: datetime
    deadline_time: datetime

    def __post_init__(self) -> None:
        if not self.season.strip():
            raise SnapshotValidationError("season must not be empty")
        if self.snapshot_gameweek <= 0:
            raise SnapshotValidationError("snapshot_gameweek must be positive")
        if self.snapshot_timestamp.tzinfo is None or self.deadline_time.tzinfo is None:
            raise SnapshotValidationError("snapshot timestamps must be timezone-aware")
        if self.snapshot_timestamp > self.deadline_time:
            raise SnapshotValidationError(
                "Snapshot was collected after the gameweek deadline and cannot "
                "be used as prediction-time data"
            )

    def as_metadata(self) -> dict[str, Any]:
        """Return JSON-compatible snapshot metadata."""
        values = asdict(self)
        values["snapshot_timestamp"] = self.snapshot_timestamp.astimezone(
            timezone.utc
        ).isoformat()
        values["snapshot_deadline_time"] = values.pop(
            "deadline_time"
        ).astimezone(timezone.utc).isoformat()
        return values


def snapshot_from_bootstrap(
    bootstrap: dict[str, Any],
    season: str,
    gameweek: int,
    collected_at: datetime | None = None,
) -> GameweekSnapshot:
    """Build snapshot metadata using the target deadline from bootstrap data."""
    events = bootstrap.get("events")
    if not isinstance(events, list):
        raise SnapshotValidationError("bootstrap events must be a list")

    event = next((row for row in events if row.get("id") == gameweek), None)
    if event is None:
        raise SnapshotValidationError(f"Gameweek {gameweek} was not found")
    raw_deadline = event.get("deadline_time")
    if not isinstance(raw_deadline, str):
        raise SnapshotValidationError(
            f"Gameweek {gameweek} has no valid deadline_time"
        )
    try:
        deadline = datetime.fromisoformat(raw_deadline.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SnapshotValidationError(
            f"Gameweek {gameweek} has an invalid deadline_time: {raw_deadline!r}"
        ) from exc

    timestamp = collected_at or datetime.now(timezone.utc)
    return GameweekSnapshot(season, gameweek, timestamp, deadline)


def attach_snapshot_metadata(
    tables: dict[str, pd.DataFrame],
    snapshot: GameweekSnapshot,
) -> dict[str, pd.DataFrame]:
    """Attach explicit snapshot identity to each normalized table."""
    result: dict[str, pd.DataFrame] = {}
    metadata = snapshot.as_metadata()
    for name, table in tables.items():
        enriched = table.copy()
        for column in (
            "season",
            "snapshot_gameweek",
            "snapshot_timestamp",
            "snapshot_deadline_time",
        ):
            enriched.insert(0, column, metadata[column])
        result[name] = enriched
    return result
