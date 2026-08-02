"""Tests for immutable web-serving bundles and application services."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_predictions.data.storage import write_json, write_parquet
from fpl_predictions.serving.bundle import (
    ServingBundle,
    ServingBundleError,
    load_serving_bundle,
)
from fpl_predictions.serving.services import (
    optimize_selection,
    rate_selection,
    sample_strong_selection,
)
from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.schemas import SquadSelection


def _rules() -> SquadRules:
    return SquadRules.from_bootstrap(_bootstrap())


def _bootstrap() -> dict[str, object]:
    return {
        "game_settings": {
            "squad_squadsize": 15,
            "squad_squadplay": 11,
            "squad_team_limit": 3,
            "squad_total_spend": 1000,
            "ui_currency_multiplier": 10,
        },
        "element_types": [
            {"id": 1, "singular_name": "Goalkeeper", "singular_name_short": "GKP", "squad_select": 2, "squad_min_play": 1, "squad_max_play": 1},
            {"id": 2, "singular_name": "Defender", "singular_name_short": "DEF", "squad_select": 5, "squad_min_play": 3, "squad_max_play": 5},
            {"id": 3, "singular_name": "Midfielder", "singular_name_short": "MID", "squad_select": 5, "squad_min_play": 2, "squad_max_play": 5},
            {"id": 4, "singular_name": "Forward", "singular_name_short": "FWD", "squad_select": 3, "squad_min_play": 1, "squad_max_play": 3},
        ],
    }


def _players() -> pd.DataFrame:
    positions = [1] * 2 + [2] * 5 + [3] * 5 + [4] * 3
    short = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    return pd.DataFrame(
        {
            "player_id": range(1, 16),
            "display_name": [f"Player {value}" for value in range(1, 16)],
            "club_id": [(value - 1) % 5 + 1 for value in range(1, 16)],
            "club_name": [f"Club {(value - 1) % 5 + 1}" for value in range(1, 16)],
            "position_id": positions,
            "position_short_name": short,
            "price": [4.0] * 15,
            "ownership_percent": [5.0] * 15,
            "status": ["a"] * 15,
        }
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": range(1, 16),
            "predicted_points_1": [float(value) for value in range(1, 16)],
        }
    )


def _references() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "goalkeeper_points": [1.0, 2.0, 3.0],
            "defence_points": [10.0, 20.0, 30.0],
            "midfield_points": [20.0, 30.0, 40.0],
            "forward_points": [15.0, 25.0, 35.0],
            "bench_points": [5.0, 10.0, 15.0],
            "captain_points": [5.0, 10.0, 15.0],
            "overall_points": [60.0, 80.0, 100.0],
        }
    )


def _selection() -> SquadSelection:
    return SquadSelection(
        player_ids=tuple(range(1, 16)),
        starting_xi=(1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15),
        bench=(2, 6, 7, 12),
        captain=15,
        vice_captain=14,
    )


def _record(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_bundle_loader_verifies_inventory_and_tables(tmp_path: Path) -> None:
    write_parquet(tmp_path / "players.parquet", _players())
    write_parquet(tmp_path / "predictions.parquet", _predictions())
    write_json(tmp_path / "bootstrap.json", _bootstrap())
    (tmp_path / "references").mkdir()
    write_parquet(tmp_path / "references/horizon-1.parquet", _references())
    write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "files": {
                "players": _record(tmp_path, "players.parquet"),
                "predictions": _record(tmp_path, "predictions.parquet"),
                "bootstrap": _record(tmp_path, "bootstrap.json"),
                "references": {
                    "1": _record(tmp_path, "references/horizon-1.parquet")
                },
                "models": [],
            },
        },
    )

    bundle = load_serving_bundle(tmp_path)

    assert bundle.horizons == (1,)
    assert len(bundle.players) == 15
    assert len(bundle.references[1]) == 3

    (tmp_path / "predictions.parquet").write_bytes(b"corrupted")
    with pytest.raises(ServingBundleError, match="size mismatch"):
        load_serving_bundle(tmp_path)


def test_serving_services_rate_and_optimize() -> None:
    bundle = ServingBundle(
        Path("."),
        {},
        _players(),
        _predictions(),
        _rules(),
        {1: _references()},
    )

    validated, projection, rating, details = rate_selection(
        bundle, _selection(), 1
    )
    optimized = optimize_selection(bundle, 1)

    assert validated.formation == "3-4-3"
    assert projection.overall_points > 0
    assert 0 <= rating.scores["overall"] < 100
    assert len(details) == 15
    assert len(optimized.selection.player_ids) == 15


@pytest.mark.skipif(
    not Path("deployment/current/manifest.json").is_file(),
    reason="serving bundle is not available",
)
def test_strong_varied_squads_are_reproducible_and_high_scoring() -> None:
    bundle = load_serving_bundle(Path("deployment/current"))
    best = optimize_selection(bundle, 3)
    first = sample_strong_selection(bundle, 3, random_seed=0)
    repeated = sample_strong_selection(bundle, 3, random_seed=0)
    second = sample_strong_selection(bundle, 3, random_seed=1)

    assert first.selection == repeated.selection
    assert first.selection != second.selection
    assert first.projection.overall_points >= 0.9 * best.projection.overall_points
    assert second.projection.overall_points >= 0.9 * best.projection.overall_points
