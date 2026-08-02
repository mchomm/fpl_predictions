"""Conventional OCR plus FPL-rule-constrained screenshot reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.ndimage import find_objects, label
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.schemas import SquadSelection
from fpl_predictions.squads.validation import validate_squad

NORMALIZED_SIZE = (471, 1024)
ROW_Y = {"GKP": 0.412, "DEF": 0.537, "MID": 0.661, "FWD": 0.786}
COUNT_X = {
    1: (0.500,),
    2: (0.333, 0.667),
    3: (0.293, 0.500, 0.707),
    4: (0.161, 0.391, 0.607, 0.833),
    5: (0.100, 0.300, 0.500, 0.700, 0.900),
}
BENCH_X = (0.161, 0.386, 0.601, 0.839)
BENCH_Y = 0.956


class ScreenshotRecognitionError(RuntimeError):
    """Raised when OCR evidence cannot support a complete legal squad."""


@dataclass(frozen=True, slots=True)
class OCRSlot:
    slot_id: str
    role: str
    expected_position: str | None
    x: float
    y: float
    readings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScreenshotRecognition:
    selection: SquadSelection
    audit: dict[str, Any]


def recognize_screenshot(
    image_path: Path,
    players: pd.DataFrame,
    rules: SquadRules,
    *,
    tesseract_command: str = "tesseract",
    screen_box: tuple[int, int, int, int] | None = None,
    corrections: Mapping[str, Any] | None = None,
    minimum_match_score: float = 0.55,
) -> ScreenshotRecognition:
    """Recognize one FPL pitch screenshot and return a validated squad."""
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not 0 <= minimum_match_score <= 1:
        raise ValueError("minimum_match_score must be between zero and one")
    executable = _resolve_executable(tesseract_command)
    normalized = _normalize_image(image_path, screen_box)
    correction_slots = _correction_slots(corrections)

    position_rules = rules.position_by_short_name
    formations = _legal_formations(rules)
    cache: dict[tuple[str, int, int], OCRSlot] = {}
    bench_slots = tuple(
        _read_slot(
            normalized,
            executable,
            f"bench:{index + 1}",
            "bench",
            None,
            x,
            BENCH_Y,
            width=0.19,
        )
        for index, x in enumerate(BENCH_X)
    )

    solutions: list[tuple[float, dict[str, Any]]] = []
    for formation in formations:
        starter_slots: list[OCRSlot] = []
        for position in ("GKP", "DEF", "MID", "FWD"):
            count = formation[position]
            for index, x in enumerate(COUNT_X[count]):
                key = (position, count, index)
                if key not in cache:
                    cache[key] = _read_slot(
                        normalized,
                        executable,
                        f"starter:{position}:{index + 1}",
                        "starter",
                        position,
                        x,
                        ROW_Y[position],
                        width=_slot_width(count),
                    )
                starter_slots.append(cache[key])
        slots = tuple(starter_slots) + bench_slots
        try:
            assignment = _assign_players(
                slots,
                players,
                rules,
                correction_slots,
            )
        except ScreenshotRecognitionError:
            continue
        solutions.append((assignment["objective"], assignment))
    if not solutions:
        raise ScreenshotRecognitionError(
            "OCR readings could not be resolved into any legal FPL squad"
        )
    _, best = max(solutions, key=lambda item: item[0])
    assigned = best["assigned"]
    low_confidence = [
        record for record in assigned if record["match_score"] < minimum_match_score
    ]

    starter_records = [row for row in assigned if row["role"] == "starter"]
    bench_records = [row for row in assigned if row["role"] == "bench"]
    marker_overrides = corrections or {}
    captain_id = _optional_positive_int(marker_overrides.get("captain"))
    vice_id = _optional_positive_int(marker_overrides.get("vice_captain"))
    detected_markers = _detect_markers(normalized, executable, starter_records)
    captain_id = captain_id or detected_markers.get("C")
    vice_id = vice_id or detected_markers.get("V")
    if captain_id is None or vice_id is None:
        missing = []
        if captain_id is None:
            missing.append("captain")
        if vice_id is None:
            missing.append("vice_captain")
        raise ScreenshotRecognitionError(
            "Could not read " + " and ".join(missing)
            + "; provide them in --corrections"
        )

    selection = SquadSelection(
        player_ids=tuple(int(row["player_id"]) for row in assigned),
        starting_xi=tuple(int(row["player_id"]) for row in starter_records),
        bench=tuple(int(row["player_id"]) for row in bench_records),
        captain=captain_id,
        vice_captain=vice_id,
        source="screenshot",
    )
    validated = validate_squad(selection, players, rules)
    audit = {
        "schema_version": 1,
        "method": "local Tesseract OCR plus exact FPL constraint optimization",
        "image": str(image_path.resolve()),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "normalized_size": list(NORMALIZED_SIZE),
        "screen_box": list(screen_box) if screen_box is not None else None,
        "formation": validated.formation,
        "total_cost": validated.total_cost,
        "minimum_match_score": minimum_match_score,
        "minimum_observed_match_score": min(
            float(row["match_score"]) for row in assigned
        ),
        "mean_match_score": float(
            np.mean([row["match_score"] for row in assigned])
        ),
        "review_required": bool(low_confidence),
        "low_confidence_slots": [
            record["slot_id"] for record in low_confidence
        ],
        "captain_detection": {
            "captain_player_id": captain_id,
            "vice_captain_player_id": vice_id,
            "detected_markers": detected_markers,
            "overridden": {
                "captain": "captain" in marker_overrides,
                "vice_captain": "vice_captain" in marker_overrides,
            },
        },
        "slots": assigned,
        "correction_format": {
            "slots": {"starter:MID:1": 154},
            "captain": 411,
            "vice_captain": 154,
        },
        "warnings": [
            "Fixture text and shirt imagery are not treated as player facts.",
            "Review alternatives for low-margin matches before rating the squad.",
        ]
        + (
            [
                "Low-confidence tentative matches require review: "
                + ", ".join(record["slot_id"] for record in low_confidence)
            ]
            if low_confidence
            else []
        ),
    }
    return ScreenshotRecognition(selection, audit)


def _normalize_image(
    path: Path,
    screen_box: tuple[int, int, int, int] | None,
) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if screen_box is not None:
        left, top, width, height = screen_box
        if left < 0 or top < 0 or width <= 0 or height <= 0:
            raise ValueError("screen_box must be non-negative x,y and positive w,h")
        if left + width > image.width or top + height > image.height:
            raise ValueError("screen_box extends beyond the image")
        image = image.crop((left, top, left + width, top + height))
    return image.resize(NORMALIZED_SIZE, Image.Resampling.LANCZOS)


def _read_slot(
    image: Image.Image,
    command: str,
    slot_id: str,
    role: str,
    expected_position: str | None,
    x_ratio: float,
    y_ratio: float,
    *,
    width: float,
) -> OCRSlot:
    x = round(image.width * x_ratio)
    y = round(image.height * y_ratio)
    crop_width = round(image.width * width)
    crop_height = round(image.height * 0.018)
    box = (
        max(x - crop_width // 2, 0),
        max(y - crop_height // 2, 0),
        min(x + crop_width // 2, image.width),
        min(y + crop_height // 2, image.height),
    )
    crop = image.crop(box)
    readings = []
    for threshold in (45, 55, 65):
        processed = _threshold_crop(crop, threshold, scale=6)
        text = _run_tesseract(command, processed, psm=7)
        if text and text not in readings:
            readings.append(text)
    return OCRSlot(
        slot_id,
        role,
        expected_position,
        x_ratio,
        y_ratio,
        tuple(readings),
    )


def _threshold_crop(image: Image.Image, threshold: int, scale: int) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    resized = grayscale.resize(
        (grayscale.width * scale, grayscale.height * scale),
        Image.Resampling.LANCZOS,
    )
    cutoff = round(255 * threshold / 100)
    return resized.point(lambda value: 255 if value >= cutoff else 0, mode="1")


def _run_tesseract(
    command: str,
    image: Image.Image,
    *,
    psm: int,
    whitelist: str | None = None,
) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        image.save(handle.name)
        arguments = [command, handle.name, "stdout", "--psm", str(psm)]
        if whitelist:
            arguments.extend(["-c", f"tessedit_char_whitelist={whitelist}"])
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    if completed.returncode != 0:
        raise ScreenshotRecognitionError(
            "Tesseract failed: " + completed.stderr.strip()
        )
    return " ".join(completed.stdout.split())


def _resolve_executable(command: str) -> str:
    if os.sep in command:
        path = Path(command)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    resolved = shutil.which(command)
    if resolved:
        return resolved
    raise ScreenshotRecognitionError(
        f"OCR executable not found: {command!r}. Install tesseract-ocr or pass "
        "--tesseract-command."
    )


def _legal_formations(rules: SquadRules) -> tuple[dict[str, int], ...]:
    counts = {"GKP": 1}
    positions = rules.position_by_short_name
    formations = []
    for defenders in range(
        positions["DEF"].minimum_starters,
        positions["DEF"].maximum_starters + 1,
    ):
        for midfielders in range(
            positions["MID"].minimum_starters,
            positions["MID"].maximum_starters + 1,
        ):
            forwards = rules.starting_size - 1 - defenders - midfielders
            if (
                positions["FWD"].minimum_starters
                <= forwards
                <= positions["FWD"].maximum_starters
            ):
                formations.append(
                    {**counts, "DEF": defenders, "MID": midfielders, "FWD": forwards}
                )
    return tuple(formations)


def _slot_width(count: int) -> float:
    return {1: 0.24, 2: 0.22, 3: 0.19, 4: 0.165, 5: 0.145}[count]


def _assign_players(
    slots: tuple[OCRSlot, ...],
    players: pd.DataFrame,
    rules: SquadRules,
    corrections: dict[str, int],
) -> dict[str, Any]:
    required = {
        "player_id",
        "display_name",
        "club_id",
        "position_short_name",
    }
    missing = required.difference(players.columns)
    if missing:
        raise ValueError("Player table is missing: " + ", ".join(sorted(missing)))
    rows = players.drop_duplicates("player_id").copy()
    records: list[dict[str, Any]] = []
    for slot_index, slot in enumerate(slots):
        pool = rows
        if slot.expected_position is not None:
            pool = pool.loc[
                pool["position_short_name"].astype(str).str.upper().eq(
                    slot.expected_position
                )
            ]
        corrected_id = corrections.get(slot.slot_id)
        if corrected_id is not None:
            pool = pool.loc[pool["player_id"].eq(corrected_id)]
            if pool.empty:
                raise ScreenshotRecognitionError(
                    f"Correction for {slot.slot_id} is not a valid candidate"
                )
        scored = []
        for row in pool.itertuples(index=False):
            score = 1.25 if corrected_id is not None else _player_match_score(
                slot.readings, row
            )
            scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], int(item[1].player_id)))
        for score, row in scored[:8]:
            records.append(
                {
                    "slot_index": slot_index,
                    "slot": slot,
                    "player_id": int(row.player_id),
                    "club_id": int(row.club_id),
                    "position": str(row.position_short_name).upper(),
                    "score": float(score),
                    "display_name": str(row.display_name),
                }
            )
    if not records:
        raise ScreenshotRecognitionError("No OCR candidates were generated")

    variable_count = len(records)
    objective = -np.array([record["score"] for record in records])
    constraints: list[tuple[dict[int, float], float, float]] = []
    for slot_index in range(len(slots)):
        indexes = [
            index
            for index, record in enumerate(records)
            if record["slot_index"] == slot_index
        ]
        constraints.append(({index: 1.0 for index in indexes}, 1.0, 1.0))
    for player_id in sorted({record["player_id"] for record in records}):
        indexes = [
            index
            for index, record in enumerate(records)
            if record["player_id"] == player_id
        ]
        constraints.append(({index: 1.0 for index in indexes}, 0.0, 1.0))
    for club_id in sorted({record["club_id"] for record in records}):
        indexes = [
            index
            for index, record in enumerate(records)
            if record["club_id"] == club_id
        ]
        constraints.append(
            ({index: 1.0 for index in indexes}, 0.0, rules.max_players_per_club)
        )
    for position in rules.positions:
        indexes = [
            index
            for index, record in enumerate(records)
            if record["position"] == position.short_name.upper()
        ]
        constraints.append(
            (
                {index: 1.0 for index in indexes},
                position.squad_count,
                position.squad_count,
            )
        )

    matrix = lil_matrix((len(constraints), variable_count), dtype=float)
    lower = np.empty(len(constraints))
    upper = np.empty(len(constraints))
    for row_index, (coefficients, low, high) in enumerate(constraints):
        for index, value in coefficients.items():
            matrix[row_index, index] = value
        lower[row_index] = low
        upper[row_index] = high
    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=int),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"mip_rel_gap": 0.0, "presolve": True},
    )
    if not result.success or result.x is None:
        raise ScreenshotRecognitionError("Candidate assignment is infeasible")
    chosen = [records[index] for index in np.flatnonzero(result.x >= 0.5)]
    chosen.sort(key=lambda record: record["slot_index"])
    assigned = []
    for record in chosen:
        slot = record["slot"]
        alternatives = [
            candidate
            for candidate in records
            if candidate["slot_index"] == record["slot_index"]
        ]
        alternatives.sort(key=lambda item: (-item["score"], item["player_id"]))
        assigned.append(
            {
                "slot_id": slot.slot_id,
                "role": slot.role,
                "expected_position": slot.expected_position,
                "x_ratio": slot.x,
                "y_ratio": slot.y,
                "ocr_readings": list(slot.readings),
                "player_id": record["player_id"],
                "display_name": record["display_name"],
                "position": record["position"],
                "match_score": record["score"],
                "corrected": slot.slot_id in corrections,
                "alternatives": [
                    {
                        "player_id": item["player_id"],
                        "display_name": item["display_name"],
                        "position": item["position"],
                        "match_score": item["score"],
                    }
                    for item in alternatives[:3]
                ],
            }
        )
    return {"objective": float(-result.fun), "assigned": assigned}


def _player_match_score(readings: tuple[str, ...], row: Any) -> float:
    aliases = {
        str(getattr(row, column))
        for column in ("display_name", "full_name", "surname")
        if hasattr(row, column) and pd.notna(getattr(row, column))
    }
    scores = [
        _text_similarity(reading, alias)
        for reading in readings
        for alias in aliases
    ]
    return max(scores, default=0.0)


def _text_similarity(observed: str, expected: str) -> float:
    left = _normalize_text(observed)
    right = _normalize_text(expected)
    if not left or not right:
        return 0.0
    full = SequenceMatcher(None, left, right).ratio()
    if right in left or left in right:
        containment = min(len(left), len(right)) / max(len(left), len(right))
        full = max(full, 0.75 + 0.25 * containment)
    tokens = re.findall(r"[a-z0-9]{3,}", _ascii_text(observed))
    token_scores = []
    for token in tokens:
        score = SequenceMatcher(None, token, right).ratio()
        if token in right or right in token:
            containment = min(len(token), len(right)) / max(len(token), len(right))
            score = max(score, 0.75 + 0.25 * containment)
        token_scores.append(score)
    token_score = max(token_scores, default=0.0)
    return max(full, token_score)


def _ascii_text(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    ).lower()


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _ascii_text(value))


def _detect_markers(
    image: Image.Image,
    command: str,
    starter_records: list[dict[str, Any]],
) -> dict[str, int]:
    pixels = np.asarray(image)
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    purple = (
        (red < 100)
        & (green < 85)
        & (blue < 125)
        & (blue > green + 5)
    )
    components, _ = label(purple)
    boxes = []
    for component_id, slices in enumerate(find_objects(components), start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        width = x_slice.stop - x_slice.start
        height = y_slice.stop - y_slice.start
        area = int((components[slices] == component_id).sum())
        if 6 <= width <= 18 and 6 <= height <= 18 and 30 <= area <= 160:
            boxes.append((x_slice, y_slice))
    detected: dict[str, int] = {}
    for record in starter_records:
        x = record["x_ratio"] * image.width
        y = record["y_ratio"] * image.height
        nearby = []
        for x_slice, y_slice in boxes:
            center_x = (x_slice.start + x_slice.stop) / 2
            center_y = (y_slice.start + y_slice.stop) / 2
            if abs(center_x - x) <= 48 and y - 100 <= center_y <= y - 45:
                distance = abs(center_x - x) + abs(center_y - (y - 72))
                nearby.append((distance, x_slice, y_slice))
        for _, x_slice, y_slice in sorted(nearby):
            padding = 3
            crop = image.crop(
                (
                    max(x_slice.start - padding, 0),
                    max(y_slice.start - padding, 0),
                    min(x_slice.stop + padding, image.width),
                    min(y_slice.stop + padding, image.height),
                )
            )
            processed = _threshold_crop(crop, 55, scale=10)
            marker = _run_tesseract(
                command, processed, psm=10, whitelist="CV"
            ).upper()
            if marker in {"C", "V"} and marker not in detected:
                detected[marker] = int(record["player_id"])
                break
    return detected


def _correction_slots(corrections: Mapping[str, Any] | None) -> dict[str, int]:
    if corrections is None:
        return {}
    raw = corrections.get("slots", {})
    if not isinstance(raw, Mapping):
        raise ValueError("corrections.slots must be an object")
    result = {}
    for slot_id, player_id in raw.items():
        parsed = _optional_positive_int(player_id)
        if parsed is None:
            raise ValueError(f"Correction for {slot_id} must be a positive ID")
        result[str(slot_id)] = parsed
    return result


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("Player corrections must be positive integer IDs")
    return value
