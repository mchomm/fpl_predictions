"""Application configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

DEFAULT_BASE_URL = "https://fantasy.premierleague.com/api/"


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings for API access and local data storage."""

    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 30.0
    data_dir: Path = Path("data")

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables, validating their values."""
        base_url = os.getenv("FPL_API_BASE_URL", DEFAULT_BASE_URL).strip()
        if not base_url:
            raise ValueError("FPL_API_BASE_URL must not be empty")

        raw_timeout = os.getenv("FPL_API_TIMEOUT_SECONDS", "30")
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise ValueError("FPL_API_TIMEOUT_SECONDS must be numeric") from exc
        if timeout <= 0:
            raise ValueError("FPL_API_TIMEOUT_SECONDS must be greater than zero")

        data_dir = Path(os.getenv("FPL_DATA_DIR", "data")).expanduser()
        return cls(base_url=base_url, timeout_seconds=timeout, data_dir=data_dir)

