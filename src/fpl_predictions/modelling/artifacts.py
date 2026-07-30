"""Versioned, schema-validated model artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fpl_predictions.data.storage import write_json
from fpl_predictions.modelling.features import FeatureSchema, prepare_features


@dataclass(slots=True)
class ModelArtifact:
    """A fitted estimator with the exact feature and provenance contract."""

    model: Any
    schema: FeatureSchema
    metadata: dict[str, Any]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict points after validating and coercing model inputs."""
        features = prepare_features(frame, self.schema)
        return np.asarray(self.model.predict(features), dtype=float)


def save_artifact(artifact: ModelArtifact, directory: Path) -> Path:
    """Atomically save a model payload and readable metadata."""
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model.joblib"
    with tempfile.NamedTemporaryFile(
        dir=directory,
        prefix=".model.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(
            {
                "model": artifact.model,
                "schema": artifact.schema.as_dict(),
                "metadata": artifact.metadata,
            },
            temporary,
        )
        temporary.replace(model_path)
        write_json(directory / "metadata.json", artifact.metadata)
        write_json(directory / "feature_schema.json", artifact.schema.as_dict())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return model_path


def load_artifact(path: Path) -> ModelArtifact:
    """Load an artifact from a model file or its containing directory."""
    model_path = path / "model.joblib" if path.is_dir() else path
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    payload = joblib.load(model_path)
    if not isinstance(payload, dict):
        raise ValueError("Model artifact payload must be a mapping")
    missing = {"model", "schema", "metadata"}.difference(payload)
    if missing:
        raise ValueError(
            "Model artifact is missing fields: " + ", ".join(sorted(missing))
        )
    return ModelArtifact(
        model=payload["model"],
        schema=FeatureSchema.from_dict(payload["schema"]),
        metadata=dict(payload["metadata"]),
    )

