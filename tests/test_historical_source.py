"""Tests for pinned external historical source handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fpl_predictions.sources.vaastav import (
    HistoricalSourceError,
    audit_season_directory,
    download_seasons,
    verify_download,
)


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def get(self, url: str, timeout: float) -> FakeResponse:
        del timeout
        return FakeResponse(f"downloaded from {url}\n".encode())


def test_audit_reports_real_schema_coverage() -> None:
    season_dir = Path(__file__).parent / "fixtures" / "vaastav" / "2023-24"

    audit = audit_season_directory(season_dir, "2023-24")

    assert audit["players"] == 2
    assert audit["gameweeks"] == 6
    assert audit["feature_coverage"]["expected_goals"] == 1.0
    assert "xP" in audit["excluded_columns"]


def test_download_is_pinned_and_checksum_verified(tmp_path) -> None:
    revision = "a" * 40
    source_root = download_seasons(
        tmp_path,
        ["2023-24"],
        revision,
        session=FakeSession(),  # type: ignore[arg-type]
    )

    manifest = verify_download(source_root, ["2023-24"])

    assert manifest["revision"] == revision
    assert len(manifest["seasons"][0]["files"]) == 3
    assert (
        source_root / "2023-24" / "gws" / "merged_gw.csv"
    ).is_file()


def test_checksum_tampering_is_detected(tmp_path) -> None:
    revision = "b" * 40
    source_root = download_seasons(
        tmp_path,
        ["2023-24"],
        revision,
        session=FakeSession(),  # type: ignore[arg-type]
    )
    target = source_root / "2023-24" / "fixtures.csv"
    target.write_text("changed", encoding="utf-8")

    with pytest.raises(HistoricalSourceError, match="Checksum mismatch"):
        verify_download(source_root, ["2023-24"])


def test_source_manifest_records_urls_and_hashes(tmp_path) -> None:
    revision = "c" * 40
    source_root = download_seasons(
        tmp_path,
        ["2023-24"],
        revision,
        session=FakeSession(),  # type: ignore[arg-type]
    )
    manifest = json.loads(
        (source_root / "source_manifest.json").read_text(encoding="utf-8")
    )

    record = manifest["seasons"][0]["files"][0]
    assert revision in record["url"]
    assert len(record["sha256"]) == 64
