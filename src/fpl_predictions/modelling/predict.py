"""Reusable player prediction service."""

from __future__ import annotations

import pandas as pd

from fpl_predictions.modelling.artifacts import ModelArtifact


def predict_players(
    artifact: ModelArtifact,
    player_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return identified player rows with predicted future FPL points."""
    identity_columns = [
        column
        for column in (
            "season",
            "snapshot_gameweek",
            "snapshot_timestamp",
            "player_id",
            "display_name",
        )
        if column in player_rows
    ]
    result = player_rows.loc[:, identity_columns].copy()
    result["prediction_horizon"] = artifact.metadata["horizon_gameweeks"]
    result["predicted_future_points"] = artifact.predict(player_rows)
    return result

