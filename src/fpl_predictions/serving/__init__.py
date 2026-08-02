"""Versioned, read-only artifacts and services for the deployed application."""

from fpl_predictions.serving.bundle import ServingBundle, load_serving_bundle

__all__ = ["ServingBundle", "load_serving_bundle"]
