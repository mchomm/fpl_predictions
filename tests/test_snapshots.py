"""Tests for deadline-aware gameweek snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from fpl_predictions.data.normalize import normalize_current_data
from fpl_predictions.data.snapshots import (
    SnapshotValidationError,
    attach_snapshot_metadata,
    snapshot_from_bootstrap,
)
from fpl_predictions.data.storage import save_current_snapshot


def test_snapshot_uses_api_deadline_and_adds_identity(
    bootstrap_payload: dict,
    fixtures_payload: list[dict],
) -> None:
    collected_at = datetime(2026, 8, 14, 12, tzinfo=timezone.utc)
    snapshot = snapshot_from_bootstrap(
        bootstrap_payload, "2026-27", 1, collected_at
    )
    tables = attach_snapshot_metadata(
        normalize_current_data(bootstrap_payload, fixtures_payload), snapshot
    )

    player = tables["players"].iloc[0]
    assert player["season"] == "2026-27"
    assert player["snapshot_gameweek"] == 1
    assert player["snapshot_timestamp"] == collected_at.isoformat()
    assert player["snapshot_deadline_time"] == "2026-08-15T10:00:00+00:00"


def test_post_deadline_snapshot_is_rejected(bootstrap_payload: dict) -> None:
    collected_at = datetime(2026, 8, 15, 10, 0, 1, tzinfo=timezone.utc)

    with pytest.raises(SnapshotValidationError, match="after the gameweek deadline"):
        snapshot_from_bootstrap(
            bootstrap_payload, "2026-27", 1, collected_at
        )


def test_snapshot_metadata_is_written_to_manifest(
    tmp_path,
    bootstrap_payload: dict,
    fixtures_payload: list[dict],
) -> None:
    collected_at = datetime(2026, 8, 14, 12, tzinfo=timezone.utc)
    snapshot = snapshot_from_bootstrap(
        bootstrap_payload, "2026-27", 1, collected_at
    )
    tables = attach_snapshot_metadata(
        normalize_current_data(bootstrap_payload, fixtures_payload), snapshot
    )

    location = save_current_snapshot(
        tmp_path,
        bootstrap_payload,
        fixtures_payload,
        tables,
        timestamp=collected_at,
        metadata=snapshot.as_metadata(),
    )

    manifest = json.loads(
        (location.processed_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["metadata"]["season"] == "2026-27"
    assert manifest["metadata"]["snapshot_gameweek"] == 1
