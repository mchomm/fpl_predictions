"""HTTP client for the official, undocumented Fantasy Premier League API."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any, TypeVar
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fpl_predictions.api.schemas import (
    ResponseShapeError,
    validate_bootstrap,
    validate_fixtures,
    validate_mapping,
)
from fpl_predictions.config import DEFAULT_BASE_URL

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class FPLAPIError(RuntimeError):
    """Raised when the FPL API cannot be reached successfully."""


class FPLResponseError(FPLAPIError):
    """Raised when the FPL API returns invalid JSON or an unexpected shape."""


class FPLClient:
    """Small FPL API client with timeouts, retries, and response validation."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if not base_url.strip():
            raise ValueError("base_url must not be empty")

        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds
        self.session = session or self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
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
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))
        session.headers.update({"User-Agent": "fpl-predictions/0.1"})
        return session

    def _get(
        self,
        path: str,
        validator: Callable[[Any], T],
    ) -> T:
        url = urljoin(self.base_url, path.lstrip("/"))
        LOGGER.info("Fetching FPL endpoint %s", url)
        try:
            response = self.session.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FPLAPIError(f"FPL request failed for {url}: {exc}") from exc

        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError as exc:
            raise FPLResponseError(f"FPL response from {url} was not valid JSON") from exc

        try:
            return validator(payload)
        except ResponseShapeError as exc:
            raise FPLResponseError(f"Invalid FPL response from {url}: {exc}") from exc

    def get_bootstrap(self) -> dict[str, Any]:
        """Return and validate ``bootstrap-static`` data."""
        return self._get("bootstrap-static/", validate_bootstrap)

    def get_fixtures(self) -> list[dict[str, Any]]:
        """Return and validate all fixtures."""
        return self._get("fixtures/", validate_fixtures)

    def get_element_summary(self, player_id: int) -> dict[str, Any]:
        """Return history and fixtures for one player."""
        return self._get(
            f"element-summary/{_positive_id(player_id, 'player_id')}/",
            lambda value: validate_mapping(value, "element-summary"),
        )

    def get_live_gameweek(self, gameweek: int) -> dict[str, Any]:
        """Return live player statistics for a gameweek."""
        return self._get(
            f"event/{_positive_id(gameweek, 'gameweek')}/live/",
            lambda value: validate_mapping(value, "event live"),
        )

    def get_manager_picks(self, manager_id: int, gameweek: int) -> dict[str, Any]:
        """Return a manager's picks for a gameweek."""
        return self._get(
            "entry/"
            f"{_positive_id(manager_id, 'manager_id')}/event/"
            f"{_positive_id(gameweek, 'gameweek')}/picks/",
            lambda value: validate_mapping(value, "manager picks"),
        )

    def get_manager_history(self, manager_id: int) -> dict[str, Any]:
        """Return a manager's season history."""
        return self._get(
            f"entry/{_positive_id(manager_id, 'manager_id')}/history/",
            lambda value: validate_mapping(value, "manager history"),
        )

    def get_manager_transfers(self, manager_id: int) -> list[dict[str, Any]]:
        """Return a manager's transfers."""
        return self._get(
            f"entry/{_positive_id(manager_id, 'manager_id')}/transfers/",
            validate_fixtures,
        )


def _positive_id(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value

