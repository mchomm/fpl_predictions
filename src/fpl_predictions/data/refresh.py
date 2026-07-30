"""Orchestration for fetching, normalizing, and storing current FPL data."""

from __future__ import annotations

import logging
from pathlib import Path

from fpl_predictions.api.client import FPLClient
from fpl_predictions.data.normalize import normalize_current_data
from fpl_predictions.data.snapshots import (
    GameweekSnapshot,
    attach_snapshot_metadata,
)
from fpl_predictions.data.storage import SnapshotLocation, save_current_snapshot

LOGGER = logging.getLogger(__name__)


def refresh_current_data(
    client: FPLClient,
    data_dir: Path,
    snapshot: GameweekSnapshot | None = None,
    bootstrap: dict | None = None,
    fixtures: list[dict] | None = None,
) -> SnapshotLocation:
    """Fetch live data and persist a reproducible raw and normalized snapshot."""
    bootstrap = bootstrap or client.get_bootstrap()
    fixtures = fixtures if fixtures is not None else client.get_fixtures()
    tables = normalize_current_data(bootstrap, fixtures)
    metadata = None
    timestamp = None
    if snapshot is not None:
        tables = attach_snapshot_metadata(tables, snapshot)
        metadata = snapshot.as_metadata()
        timestamp = snapshot.snapshot_timestamp
    location = save_current_snapshot(
        data_dir,
        bootstrap,
        fixtures,
        tables,
        timestamp=timestamp,
        metadata=metadata,
    )
    LOGGER.info(
        "Saved FPL snapshot %s to %s",
        location.snapshot_id,
        location.processed_dir,
    )
    return location
