"""Lightweight runtime validation for undocumented FPL API responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ResponseShapeError(ValueError):
    """Raised when an API response does not have the expected structure."""


BOOTSTRAP_COLLECTIONS = ("elements", "teams", "events", "element_types")


def validate_bootstrap(payload: Any) -> dict[str, Any]:
    """Validate the collections required to normalize bootstrap data."""
    if not isinstance(payload, Mapping):
        raise ResponseShapeError("bootstrap-static response must be a JSON object")

    missing = [key for key in BOOTSTRAP_COLLECTIONS if key not in payload]
    if missing:
        raise ResponseShapeError(
            "bootstrap-static response is missing required keys: "
            + ", ".join(missing)
        )

    for key in BOOTSTRAP_COLLECTIONS:
        value = payload[key]
        if not isinstance(value, list):
            raise ResponseShapeError(f"bootstrap-static field {key!r} must be a list")
        if any(not isinstance(row, Mapping) for row in value):
            raise ResponseShapeError(
                f"bootstrap-static field {key!r} must contain only objects"
            )

    return dict(payload)


def validate_fixtures(payload: Any) -> list[dict[str, Any]]:
    """Validate the top-level fixtures collection."""
    if (
        not isinstance(payload, Sequence)
        or isinstance(payload, (str, bytes, bytearray))
        or any(not isinstance(row, Mapping) for row in payload)
    ):
        raise ResponseShapeError("fixtures response must be a JSON list of objects")
    return [dict(row) for row in payload]


def validate_mapping(payload: Any, endpoint: str) -> dict[str, Any]:
    """Validate an endpoint whose response must be a JSON object."""
    if not isinstance(payload, Mapping):
        raise ResponseShapeError(f"{endpoint} response must be a JSON object")
    return dict(payload)

