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
from scipy.ndimage import find_objects, label, maximum_filter, uniform_filter
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from fpl_predictions.squads.rules import SquadRules
from fpl_predictions.squads.schemas import SquadSelection
from fpl_predictions.squads.validation import validate_squad

NORMALIZED_WIDTH = 471


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
class NameplateCandidate:
    """A visually detected player nameplate."""

    x: float
    y: float
    width: float
    height: float
    visual_score: float


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

    starter_plates, bench_plates, layout_audit = _detect_nameplates(normalized)
    slots = tuple(
        _read_nameplate(
            normalized,
            executable,
            f"starter:{index + 1}",
            "starter",
            plate,
        )
        for index, plate in enumerate(starter_plates)
    ) + tuple(
        _read_nameplate(
            normalized,
            executable,
            f"bench:{index + 1}",
            "bench",
            plate,
        )
        for index, plate in enumerate(bench_plates)
    )
    best = _assign_players(slots, players, rules, correction_slots)
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
        "method": (
            "dynamic nameplate detection, local Tesseract OCR, and exact FPL "
            "constraint optimization"
        ),
        "image": str(image_path.resolve()),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "normalized_size": list(normalized.size),
        "screen_box": list(screen_box) if screen_box is not None else None,
        "layout_detection": layout_audit,
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
            "slots": {"starter:1": 154},
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
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if screen_box is not None:
        left, top, width, height = screen_box
        if left < 0 or top < 0 or width <= 0 or height <= 0:
            raise ValueError("screen_box must be non-negative x,y and positive w,h")
        if left + width > image.width or top + height > image.height:
            raise ValueError("screen_box extends beyond the image")
        image = image.crop((left, top, left + width, top + height))
    else:
        pixels = np.asarray(image)
        red = pixels[:, :, 0].astype(float)
        green = pixels[:, :, 1].astype(float)
        blue = pixels[:, :, 2].astype(float)
        pitch = (
            (green > 65)
            & (green > red * 1.12)
            & (green > blue * 1.03)
        )
        components, count = label(pitch)
        if count:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            pitch_id = int(sizes.argmax())
            _, pitch_x = np.where(components == pitch_id)
            detected_width = int(pitch_x.max() - pitch_x.min() + 1)
            if detected_width < image.width * 0.88:
                padding = round(detected_width * 0.025)
                left = max(0, int(pitch_x.min()) - padding)
                right = min(image.width, int(pitch_x.max()) + padding + 1)
                image = image.crop((left, 0, right, image.height))
    if image.width == NORMALIZED_WIDTH:
        return image
    height = max(1, round(image.height * NORMALIZED_WIDTH / image.width))
    return image.resize((NORMALIZED_WIDTH, height), Image.Resampling.LANCZOS)


def _detect_nameplates(
    image: Image.Image,
) -> tuple[tuple[NameplateCandidate, ...], tuple[NameplateCandidate, ...], dict[str, Any]]:
    """Find 11 starter and four bench nameplates without assuming a formation."""
    pixels = np.asarray(image)
    red = pixels[:, :, 0].astype(float)
    green = pixels[:, :, 1].astype(float)
    blue = pixels[:, :, 2].astype(float)
    pitch_mask = (
        (green > 65)
        & (green > red * 1.12)
        & (green > blue * 1.03)
    )
    components, count = label(pitch_mask)
    if count == 0:
        raise ScreenshotRecognitionError(
            "Could not locate the FPL pitch; upload the full Pick Team screen"
        )
    component_sizes = np.bincount(components.ravel())
    component_sizes[0] = 0
    pitch_id = int(component_sizes.argmax())
    pitch_y, pitch_x = np.where(components == pitch_id)
    pitch_width = int(pitch_x.max() - pitch_x.min() + 1)
    if pitch_width < max(120, round(image.width * 0.25)):
        raise ScreenshotRecognitionError(
            "The detected FPL pitch is too small for reliable player OCR"
        )

    plate_width = max(34, round(pitch_width * 0.16))
    plate_height = max(16, round(pitch_width * 0.072))
    lower_top = max(0, int(pitch_y.min() + pitch_width * 0.07))
    lower = pixels[lower_top:]
    lower_brightness = lower.min(axis=2)
    bright_threshold = float(
        np.clip(np.percentile(lower_brightness, 78), 145, 215)
    )
    minimum = pixels.min(axis=2).astype(float)
    maximum = pixels.max(axis=2).astype(float)
    neutral = (
        (minimum >= bright_threshold)
        & ((maximum - minimum) <= np.maximum(38, maximum * 0.20))
    )
    luminance = red * 0.299 + green * 0.587 + blue * 0.114
    dark = luminance < min(165.0, bright_threshold * 0.76)

    white_density = uniform_filter(
        neutral.astype(float), size=(plate_height, plate_width), mode="constant"
    )
    dark_density = uniform_filter(
        dark.astype(float), size=(plate_height, plate_width), mode="constant"
    )
    # Player labels are bright, mostly neutral rectangles with roughly ten per
    # cent dark glyph pixels. Penalizing blank white areas prevents the page
    # background and shirt highlights from winning.
    score = white_density - 3.0 * np.abs(dark_density - 0.10)
    allowed = np.zeros(score.shape, dtype=bool)
    x_padding = round(pitch_width * 0.06)
    x_start = max(plate_width // 2, int(pitch_x.min()) - x_padding)
    x_stop = min(
        image.width - plate_width // 2,
        int(pitch_x.max()) + x_padding + 1,
    )
    y_start = max(plate_height // 2, lower_top)
    y_stop = image.height - plate_height // 2
    allowed[y_start:y_stop, x_start:x_stop] = True
    score[
        (~allowed)
        | (white_density < 0.48)
        | (dark_density < 0.035)
        | (dark_density > 0.24)
    ] = -1.0

    local_maximum = maximum_filter(
        score,
        size=(max(5, plate_height // 2), max(7, plate_width // 2)),
        mode="constant",
        cval=-1.0,
    )
    maxima_y, maxima_x = np.where(
        (score == local_maximum) & (score >= 0.22)
    )
    ranked = sorted(
        (
            NameplateCandidate(
                x=float(x),
                y=float(y),
                width=float(plate_width),
                height=float(plate_height),
                visual_score=float(score[y, x]),
            )
            for y, x in zip(maxima_y, maxima_x, strict=True)
        ),
        key=lambda candidate: (-candidate.visual_score, candidate.y, candidate.x),
    )
    candidates: list[NameplateCandidate] = []
    for candidate in ranked:
        if any(
            abs(candidate.x - chosen.x) < plate_width * 0.65
            and abs(candidate.y - chosen.y) < plate_height * 0.70
            for chosen in candidates
        ):
            continue
        candidates.append(candidate)
        if len(candidates) >= 60:
            break

    rows = _cluster_nameplate_rows(candidates, plate_height)
    bench_options = [row for row in rows if len(row) >= 4]
    if not bench_options:
        raise ScreenshotRecognitionError(
            "Could not find the row of four substitute nameplates"
        )
    pitch_bottom = float(pitch_y.max())
    bench_row = min(
        bench_options,
        key=lambda row: abs(
            float(np.median([candidate.y for candidate in row])) - pitch_bottom
        ),
    )
    # Bench name and fixture text can merge into one candidate row. The four
    # maxima nearest the bottom of the detected pitch consistently identify
    # the name line; translate that line back to the plate centre expected by
    # _read_nameplate.
    bench_name_candidates = sorted(
        bench_row, key=lambda candidate: abs(candidate.y - pitch_bottom)
    )[:4]
    bench_name_y = float(
        np.median([candidate.y for candidate in bench_name_candidates])
    )
    bench_y = bench_name_y + plate_height * 0.17
    # The substitute tray always contains four columns, but its absolute
    # coordinates and pixel size vary by device. Derive those columns from the
    # detected pitch width instead of trusting local maxima that can drift into
    # the pale bench background near the bottom of the image.
    bench_centers = np.linspace(
        float(pitch_x.min()) + pitch_width * 0.16,
        float(pitch_x.max()) - pitch_width * 0.16,
        4,
    )
    bench = [
        NameplateCandidate(
            x=float(center),
            y=bench_y,
            width=float(plate_width),
            height=float(plate_height),
            visual_score=max(
                (
                    candidate.visual_score
                    for candidate in bench_row
                    if abs(candidate.x - center) <= plate_width
                ),
                default=0.0,
            ),
        )
        for center in bench_centers
    ]

    starter_pool: list[NameplateCandidate] = []
    for row in rows:
        row_y = float(np.mean([candidate.y for candidate in row]))
        if row_y >= bench_y - plate_height * 1.1:
            continue
        starter_pool.extend(
            sorted(row, key=lambda item: item.visual_score, reverse=True)[:5]
        )
    selected_starters = sorted(
        starter_pool, key=lambda item: item.visual_score, reverse=True
    )[:11]
    starters = [
        candidate
        for row in _cluster_nameplate_rows(selected_starters, plate_height)
        for candidate in sorted(row, key=lambda item: item.x)
    ]
    if len(starters) != 11 or len(bench) != 4:
        raise ScreenshotRecognitionError(
            f"Detected {len(starters)} starter and {len(bench)} substitute "
            "nameplates; expected 11 and 4. Candidate row counts were "
            f"{[len(row) for row in rows]}"
        )

    return (
        tuple(starters),
        tuple(bench),
        {
            "method": "adaptive bright-nameplate detection",
            "pitch_bounds": [
                int(pitch_x.min()),
                int(pitch_y.min()),
                int(pitch_x.max() + 1),
                int(pitch_y.max() + 1),
            ],
            "plate_size": [plate_width, plate_height],
            "bright_threshold": bright_threshold,
            "candidate_count": len(candidates),
            "candidate_centers": [
                [round(item.x, 1), round(item.y, 1), round(item.visual_score, 4)]
                for item in sorted(candidates, key=lambda item: (item.y, item.x))
            ],
            "detected_row_counts": [len(row) for row in rows],
            "starter_centers": [[round(item.x, 1), round(item.y, 1)] for item in starters],
            "bench_centers": [[round(item.x, 1), round(item.y, 1)] for item in bench],
        },
    )


def _cluster_nameplate_rows(
    candidates: list[NameplateCandidate],
    plate_height: int,
) -> list[list[NameplateCandidate]]:
    """Cluster candidates into visual rows while tolerating mild perspective."""
    rows: list[list[NameplateCandidate]] = []
    for candidate in sorted(candidates, key=lambda item: (item.y, item.x)):
        matching = next(
            (
                row
                for row in rows
                if abs(candidate.y - np.median([item.y for item in row]))
                <= plate_height * 0.85
            ),
            None,
        )
        if matching is None:
            rows.append([candidate])
        else:
            matching.append(candidate)
    return rows


def _read_nameplate(
    image: Image.Image,
    command: str,
    slot_id: str,
    role: str,
    plate: NameplateCandidate,
) -> OCRSlot:
    """OCR the name line within one detected nameplate."""
    crop_width = round(plate.width * 1.12)
    crop_height = max(12, round(plate.height * 0.62))
    center_y = round(plate.y - plate.height * 0.17)
    center_x = round(plate.x)
    box = (
        max(center_x - crop_width // 2, 0),
        max(center_y - crop_height // 2, 0),
        min(center_x + crop_width // 2, image.width),
        min(center_y + crop_height // 2, image.height),
    )
    crop = image.crop(box)
    readings = []
    grayscale = ImageOps.grayscale(crop)
    raw = grayscale.resize(
        (grayscale.width * 6, grayscale.height * 6),
        Image.Resampling.LANCZOS,
    )
    for psm in (7, 11):
        text = _run_tesseract(command, raw, psm=psm)
        if text and text not in readings:
            readings.append(text)
    for threshold in (40, 50, 60, 70):
        processed = _threshold_crop(crop, threshold, scale=6)
        text = _run_tesseract(command, processed, psm=7)
        if text and text not in readings:
            readings.append(text)
    return OCRSlot(
        slot_id=slot_id,
        role=role,
        expected_position=None,
        x=plate.x / image.width,
        y=plate.y / image.height,
        readings=tuple(readings),
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
        starter_indexes = [
            index
            for index, record in enumerate(records)
            if record["position"] == position.short_name.upper()
            and record["slot"].role == "starter"
        ]
        constraints.append(
            (
                {index: 1.0 for index in starter_indexes},
                position.minimum_starters,
                position.maximum_starters,
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
        max((_text_similarity(reading, alias) for alias in aliases), default=0.0)
        for reading in readings
    ]
    if not scores:
        return 0.0
    # A single OCR pass can hallucinate a different short player name. Blend
    # the best pass with cross-pass consensus so one accidental exact token
    # cannot beat a consistently near-exact reading of the real name.
    return float(0.55 * max(scores) + 0.45 * np.median(scores))


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
            for threshold in (25, 40, 55, 65):
                processed = _threshold_crop(crop, threshold, scale=10)
                marker = _run_tesseract(
                    command, processed, psm=10, whitelist="CV"
                ).upper()
                if marker in {"C", "V"} and marker not in detected:
                    detected[marker] = int(record["player_id"])
                    break
            if int(record["player_id"]) in detected.values():
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
