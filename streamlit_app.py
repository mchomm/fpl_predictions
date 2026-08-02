"""Interactive FPL screenshot rating and optimization application."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import streamlit as st

from fpl_predictions.screenshot.recognition import (
    ScreenshotRecognitionError,
    recognize_screenshot,
)
from fpl_predictions.serving.bundle import (
    ServingBundleError,
    load_serving_bundle,
)
from fpl_predictions.serving.services import (
    optimize_selection,
    player_label_lookup,
    rate_selection,
)
from fpl_predictions.squads.schemas import SquadSelection
from fpl_predictions.squads.validation import SquadValidationError, validate_squad


BUNDLE_ROOT = Path(os.getenv("FPL_SERVING_BUNDLE", "deployment/current"))


@st.cache_resource(show_spinner="Loading current FPL models and data…")
def _bundle() -> Any:
    return load_serving_bundle(BUNDLE_ROOT)


def _set_selection(selection: SquadSelection, audit: dict[str, Any] | None = None) -> None:
    st.session_state["selection"] = selection.as_dict()
    st.session_state["recognition_audit"] = audit
    st.session_state.pop("rating_result", None)
    st.session_state.pop("optimization_result", None)


def _selection() -> SquadSelection | None:
    payload = st.session_state.get("selection")
    return SquadSelection.from_mapping(payload) if isinstance(payload, dict) else None


def _player_table(bundle: Any, selection: SquadSelection) -> pd.DataFrame:
    indexed = bundle.players.set_index("player_id")
    records = []
    for role, values in (
        ("Starter", selection.starting_xi),
        ("Bench", selection.bench),
    ):
        for order, player_id in enumerate(values, start=1):
            row = indexed.loc[player_id]
            records.append(
                {
                    "Role": role,
                    "Order": order,
                    "Player": row["display_name"],
                    "Club": row["club_name"],
                    "Position": row["position_short_name"],
                    "Price": float(row["price"]),
                    "Captaincy": (
                        "C"
                        if player_id == selection.captain
                        else "V"
                        if player_id == selection.vice_captain
                        else ""
                    ),
                    "ID": int(player_id),
                }
            )
    return pd.DataFrame(records)


def _render_editor(bundle: Any, selection: SquadSelection) -> SquadSelection:
    labels, _ = player_label_lookup(bundle.players)
    indexed = bundle.players.set_index("player_id")
    st.subheader("Review and edit squad")
    st.caption(
        "Every OCR match remains editable. Player IDs stay attached to names, "
        "so corrections are validated against the current database."
    )
    edited_starters = []
    edited_bench = []
    tabs = st.tabs(["Starting XI", "Bench and order"])
    with tabs[0]:
        columns = st.columns(3)
        for index, player_id in enumerate(selection.starting_xi):
            position = str(indexed.loc[player_id, "position_short_name"])
            options = bundle.players.loc[
                bundle.players["position_short_name"].eq(position), "player_id"
            ].astype(int).tolist()
            options.sort(key=lambda value: labels[value])
            chosen = columns[index % 3].selectbox(
                f"Starter {index + 1} · {position}",
                options,
                index=options.index(player_id),
                format_func=lambda value, mapping=labels: mapping[value],
                key=f"starter_editor_{index}",
            )
            edited_starters.append(int(chosen))
    with tabs[1]:
        columns = st.columns(2)
        for index, player_id in enumerate(selection.bench):
            position = str(indexed.loc[player_id, "position_short_name"])
            options = bundle.players.loc[
                bundle.players["position_short_name"].eq(position), "player_id"
            ].astype(int).tolist()
            options.sort(key=lambda value: labels[value])
            chosen = columns[index % 2].selectbox(
                f"Bench slot {index + 1} · {position}",
                options,
                index=options.index(player_id),
                format_func=lambda value, mapping=labels: mapping[value],
                key=f"bench_editor_{index}",
            )
            edited_bench.append(int(chosen))

    starter_options = list(dict.fromkeys(edited_starters))
    cap_col, vice_col = st.columns(2)
    captain = cap_col.selectbox(
        "Captain",
        starter_options,
        index=(
            starter_options.index(selection.captain)
            if selection.captain in starter_options
            else 0
        ),
        format_func=lambda value: labels[value],
        key="captain_editor",
    )
    vice_default = (
        starter_options.index(selection.vice_captain)
        if selection.vice_captain in starter_options
        else min(1, len(starter_options) - 1)
    )
    vice = vice_col.selectbox(
        "Vice-captain",
        starter_options,
        index=vice_default,
        format_func=lambda value: labels[value],
        key="vice_editor",
    )
    edited = SquadSelection(
        player_ids=tuple(edited_starters + edited_bench),
        starting_xi=tuple(edited_starters),
        bench=tuple(edited_bench),
        captain=int(captain),
        vice_captain=int(vice),
        source=selection.source,
        bank=selection.bank,
        budget_limit=selection.budget_limit,
        purchase_prices=selection.purchase_prices,
        selling_prices=selection.selling_prices,
    )
    if st.button("Apply and validate edits", type="primary"):
        try:
            validated = validate_squad(edited, bundle.players, bundle.rules)
        except SquadValidationError as exc:
            st.error(str(exc))
        else:
            _set_selection(edited, st.session_state.get("recognition_audit"))
            st.success(
                f"Legal {validated.formation} squad · £{validated.total_cost:.1f}m"
            )
            st.rerun()
    return edited


def _recognition_table(audit: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Slot": row["slot_id"],
            "OCR": " / ".join(row["ocr_readings"]),
            "Selected": row["display_name"],
            "Confidence": f"{100 * row['match_score']:.1f}%",
            "Alternatives": ", ".join(
                item["display_name"] for item in row["alternatives"][1:3]
            ),
        }
        for row in audit["slots"]
    )


st.set_page_config(
    page_title="FPL Squad Lab",
    page_icon="⚽",
    layout="wide",
)
st.title("FPL Squad Lab")
st.caption("Screenshot recognition, model-derived ratings, and legal squad optimization")

try:
    bundle = _bundle()
except (ServingBundleError, FileNotFoundError, OSError, ValueError) as exc:
    st.error(f"The deployed serving bundle could not be loaded: {exc}")
    st.stop()

manifest = bundle.manifest
with st.sidebar:
    st.header("Current model")
    st.write(f"Season: **{manifest.get('season') or 'unknown'}**")
    st.write(f"Snapshot GW: **{manifest.get('snapshot_gameweek') or 'unknown'}**")
    st.write(f"Updated: **{manifest.get('prediction_created_at_utc') or 'unknown'}**")
    horizon = st.selectbox(
        "Prediction horizon",
        bundle.horizons,
        index=(bundle.horizons.index(3) if 3 in bundle.horizons else 0),
        format_func=lambda value: f"Next {value} Gameweek{'s' if value != 1 else ''}",
    )
    st.caption("Ratings and recommendations use the same model version.")

upload_col, action_col = st.columns([2, 1])
with upload_col:
    uploaded = st.file_uploader(
        "Upload an FPL Pick Team screenshot",
        type=["png", "jpg", "jpeg"],
        help="Clean native screenshots work best. Uploaded images are processed temporarily.",
    )
with action_col:
    st.write("")
    st.write("")
    recognize_clicked = st.button(
        "Recognize screenshot",
        type="primary",
        disabled=uploaded is None,
        width="stretch",
    )

if recognize_clicked and uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower() or ".png"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(uploaded.getvalue())
            temporary_path = Path(handle.name)
        with st.spinner("Reading player labels and solving the legal squad…"):
            recognition = recognize_screenshot(
                temporary_path,
                bundle.players,
                bundle.rules,
            )
        _set_selection(recognition.selection, recognition.audit)
        st.success(
            f"Recognized {recognition.audit['formation']} · "
            f"mean match {100 * recognition.audit['mean_match_score']:.1f}%"
        )
        st.rerun()
    except (ScreenshotRecognitionError, OSError, ValueError) as exc:
        st.error(str(exc))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

selection = _selection()
if selection is None:
    st.info("Upload a screenshot, or generate the model's best legal squad below.")
    if st.button("Generate best possible squad"):
        with st.spinner("Solving the exact squad optimization…"):
            optimized = optimize_selection(bundle, horizon)
        _set_selection(optimized.selection)
        st.session_state["optimization_result"] = optimized.as_dict(
            bundle.players, bundle.predictions
        )
        st.rerun()
    st.stop()

audit = st.session_state.get("recognition_audit")
if isinstance(audit, dict):
    if audit.get("review_required"):
        st.warning(
            "Review required for: " + ", ".join(audit["low_confidence_slots"])
        )
    with st.expander("OCR evidence and alternatives"):
        st.dataframe(_recognition_table(audit), hide_index=True, width="stretch")

current_tab, rating_tab, improve_tab, model_tab = st.tabs(
    ["Squad", "Rating", "Improve", "Model provenance"]
)

with current_tab:
    try:
        current_validated = validate_squad(selection, bundle.players, bundle.rules)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Formation", current_validated.formation)
        metric_cols[1].metric("Squad cost", f"£{current_validated.total_cost:.1f}m")
        metric_cols[2].metric("Bank", "Unknown" if selection.bank is None else f"£{selection.bank:.1f}m")
        st.dataframe(_player_table(bundle, selection), hide_index=True, width="stretch")
    except SquadValidationError as exc:
        st.error(str(exc))
    _render_editor(bundle, selection)

with rating_tab:
    if st.button("Rate this squad", type="primary"):
        try:
            with st.spinner("Applying the point model and reference calibration…"):
                validated, projection, rating, details = rate_selection(
                    bundle, selection, horizon
                )
            st.session_state["rating_result"] = {
                "validation": {
                    "formation": validated.formation,
                    "total_cost": validated.total_cost,
                },
                "projection": projection.as_dict(),
                "rating": rating.as_dict(),
                "players": details,
            }
        except (SquadValidationError, ValueError) as exc:
            st.error(str(exc))
    rating_result = st.session_state.get("rating_result")
    if isinstance(rating_result, dict):
        scores = rating_result["rating"]["scores"]
        columns = st.columns(len(scores))
        for column, (name, score) in zip(columns, scores.items(), strict=True):
            column.metric(name.title(), f"{score:.1f}")
        projection = rating_result["projection"]
        st.write(
            f"Projected {horizon}-GW overall points: "
            f"**{projection['overall_points']:.2f}**"
        )
        st.caption(rating_result["rating"]["interpretation"])
        st.dataframe(pd.DataFrame(rating_result["players"]), hide_index=True, width="stretch")
        st.download_button(
            "Download rating JSON",
            json.dumps(rating_result, indent=2),
            file_name="fpl-rating.json",
            mime="application/json",
        )

with improve_tab:
    control_cols = st.columns(3)
    max_transfers = control_cols[0].slider("Maximum transfers", 1, 5, 3)
    preseason = control_cols[1].checkbox(
        "Preseason / transfers are free",
        value=bool(manifest.get("snapshot_gameweek") == 1),
    )
    free_transfers = control_cols[2].number_input(
        "Available free transfers",
        min_value=0,
        max_value=5,
        value=max_transfers if preseason else 1,
        disabled=preseason,
    )
    improve_col, rebuild_col = st.columns(2)
    improve = improve_col.button("Suggest transfers", type="primary", width="stretch")
    rebuild = rebuild_col.button("Generate best squad from scratch", width="stretch")
    if improve or rebuild:
        try:
            with st.spinner("Solving exact FPL constraints…"):
                optimized = optimize_selection(
                    bundle,
                    horizon,
                    current_squad=(None if rebuild else selection),
                    max_transfers=(None if rebuild else max_transfers),
                    free_transfers=(
                        1
                        if rebuild
                        else max_transfers
                        if preseason
                        else int(free_transfers)
                    ),
                )
            st.session_state["optimization_result"] = optimized.as_dict(
                bundle.players, bundle.predictions
            )
        except (RuntimeError, SquadValidationError, ValueError) as exc:
            st.error(str(exc))
    optimization = st.session_state.get("optimization_result")
    if isinstance(optimization, dict):
        projection = optimization["projection"]
        metric_cols = st.columns(4)
        metric_cols[0].metric("Formation", projection["formation"])
        metric_cols[1].metric("Projected points", f"{projection['overall_points']:.2f}")
        metric_cols[2].metric(
            "Gross gain",
            "New squad"
            if optimization["projected_points_gain"] is None
            else f"{optimization['projected_points_gain']:+.2f}",
        )
        metric_cols[3].metric(
            "Net gain",
            "New squad"
            if optimization["net_projected_points_gain"] is None
            else f"{optimization['net_projected_points_gain']:+.2f}",
        )
        outgoing = optimization["transfers_out"]
        incoming = optimization["transfers_in"]
        if incoming:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Transfer out": [row["display_name"] for row in outgoing],
                        "Transfer in": [row["display_name"] for row in incoming],
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        st.dataframe(
            pd.DataFrame(optimization["selected_players"]),
            hide_index=True,
            width="stretch",
        )
        if st.button("Use this optimized squad"):
            _set_selection(SquadSelection.from_mapping(optimization["selection"]))
            st.rerun()
        st.download_button(
            "Download optimization JSON",
            json.dumps(optimization, indent=2),
            file_name="fpl-optimization.json",
            mime="application/json",
        )

with model_tab:
    st.json(
        {
            "season": manifest.get("season"),
            "snapshot_gameweek": manifest.get("snapshot_gameweek"),
            "snapshot_timestamp": manifest.get("snapshot_timestamp"),
            "prediction_created_at_utc": manifest.get("prediction_created_at_utc"),
            "model_runs": manifest.get("model_runs"),
            "horizons": manifest.get("horizons"),
            "reference_configuration": manifest.get("reference_configuration"),
            "runtime_policy": manifest.get("runtime_policy"),
        }
    )
    st.caption(
        "The deployed bundle includes the checksummed model artifacts that "
        "produced these predictions. Training data and uploaded screenshots are not persisted."
    )
