"""Tests for transparent point and expected-minutes baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from fpl_predictions.modelling.baselines import (
    RecentWithModelFallbackRegressor,
)


def test_recent_minutes_uses_learned_fallback_only_for_cold_starts() -> None:
    frame = pd.DataFrame(
        {
            "recent_minutes_mean_3": [80.0, np.nan, 45.0],
            "price": [8.0, 6.0, 5.0],
        }
    )
    model = RecentWithModelFallbackRegressor(
        "recent_minutes_mean_3",
        horizon=3,
        fallback_estimator=DummyRegressor(strategy="constant", constant=99.0),
    )
    model.fit(frame, pd.Series([240.0, 90.0, 135.0]))

    assert model.predict(frame).tolist() == [240.0, 99.0, 135.0]
