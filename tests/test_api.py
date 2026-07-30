"""Tests for FPL HTTP and response-shape handling."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from fpl_predictions.api.client import FPLAPIError, FPLClient, FPLResponseError


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        status_error: requests.RequestException | None = None,
    ) -> None:
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self) -> None:
        if self.status_error:
            raise self.status_error

    def json(self) -> Any:
        return self.payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, float]] = []

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.calls.append((url, timeout))
        return self.response


def test_bootstrap_uses_configured_url_and_timeout(bootstrap_payload: dict) -> None:
    session = FakeSession(FakeResponse(bootstrap_payload))
    client = FPLClient(
        "https://example.test/custom-api",
        timeout_seconds=4.5,
        session=session,  # type: ignore[arg-type]
    )

    assert client.get_bootstrap()["elements"][0]["id"] == 101
    assert session.calls == [
        ("https://example.test/custom-api/bootstrap-static/", 4.5)
    ]


def test_bootstrap_rejects_missing_collection(bootstrap_payload: dict) -> None:
    del bootstrap_payload["element_types"]
    client = FPLClient(
        session=FakeSession(FakeResponse(bootstrap_payload))  # type: ignore[arg-type]
    )

    with pytest.raises(FPLResponseError, match="element_types"):
        client.get_bootstrap()


def test_http_error_has_endpoint_context() -> None:
    error = requests.HTTPError("503 Server Error")
    client = FPLClient(
        session=FakeSession(FakeResponse(None, status_error=error))  # type: ignore[arg-type]
    )

    with pytest.raises(FPLAPIError, match="bootstrap-static"):
        client.get_bootstrap()


@pytest.mark.parametrize("value", [0, -1, True, "4"])
def test_dynamic_endpoints_require_positive_integer_ids(value: Any) -> None:
    client = FPLClient(session=FakeSession(FakeResponse({})))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="positive integer"):
        client.get_element_summary(value)  # type: ignore[arg-type]
