"""Shared saved API responses for tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_json(name: str) -> Any:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def bootstrap_payload() -> dict[str, Any]:
    """Return a compact saved bootstrap response."""
    return _load_json("bootstrap_static.json")


@pytest.fixture
def fixtures_payload() -> list[dict[str, Any]]:
    """Return a compact saved fixtures response."""
    return _load_json("fixtures.json")

