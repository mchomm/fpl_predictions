"""CLI for local OCR reconstruction of an FPL squad screenshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import duckdb
import pandas as pd

from fpl_predictions.config import Settings
from fpl_predictions.data.storage import write_json
from fpl_predictions.squad_cli import _latest_predictions, _load_snapshot_context
from fpl_predictions.screenshot.recognition import (
    ScreenshotRecognitionError,
    recognize_screenshot,
)
from fpl_predictions.squads.rules import SquadRules


def main(argv: Sequence[str] | None = None) -> int:
    """Recognize a screenshot into editable audit and runnable squad JSON."""
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="fpl-recognize-screenshot",
        description="Read an FPL Pick Team screenshot with local OCR.",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--tesseract-command", default="tesseract")
    parser.add_argument(
        "--screen-box",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Crop a photographed phone screen before OCR.",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        help="Optional JSON containing slot player IDs and captain overrides.",
    )
    parser.add_argument("--minimum-match-score", type=float, default=0.55)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args(argv)

    try:
        prediction_path = args.predictions or _latest_predictions(
            args.data_dir.parent / "outputs" / "predictions"
        )
        predictions = pd.read_parquet(prediction_path)
        players, bootstrap, snapshot = _load_snapshot_context(
            args.data_dir, args.database, predictions
        )
        rules = SquadRules.from_bootstrap(bootstrap)
        corrections = None
        if args.corrections is not None:
            corrections = json.loads(args.corrections.read_text(encoding="utf-8"))
            if not isinstance(corrections, dict):
                raise ValueError("Corrections JSON must contain one object")
        result = recognize_screenshot(
            args.image,
            players,
            rules,
            tesseract_command=args.tesseract_command,
            screen_box=(tuple(args.screen_box) if args.screen_box else None),
            corrections=corrections,
            minimum_match_score=args.minimum_match_score,
        )
        output = args.output or args.image.with_suffix(".squad.json")
        audit_output = args.audit_output or output.with_name(
            output.stem + ".recognition.json"
        )
        audit = {
            **result.audit,
            "snapshot": snapshot,
            "prediction_file": str(prediction_path.resolve()),
            "squad_output": str(output.resolve()),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result.selection.as_dict())
        write_json(audit_output, audit)
    except (
        duckdb.Error,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        ScreenshotRecognitionError,
        ValueError,
    ) as exc:
        parser.error(str(exc))

    print(f"Formation: {audit['formation']}")
    print(f"Squad cost: £{audit['total_cost']:.1f}m")
    print(f"Mean OCR match: {100 * audit['mean_match_score']:.1f}%")
    if audit["review_required"]:
        print(
            "Review required: " + ", ".join(audit["low_confidence_slots"])
        )
    print(f"Squad: {output}")
    print(f"Recognition audit: {audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
