"""Interactive FPL squad recognition, rating, and optimization application."""

from __future__ import annotations

from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

# Community Cloud runs this entrypoint from the repository root. Add the
# src-layout package directly so deployment does not need an editable install.
REPOSITORY_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fpl_predictions.screenshot.recognition import (  # noqa: E402
    ScreenshotRecognitionError,
    recognize_screenshot,
)
from fpl_predictions.serving.bundle import (  # noqa: E402
    ServingBundleError,
    load_serving_bundle,
)
from fpl_predictions.serving.presentation import (  # noqa: E402
    POSITION_ORDER,
    build_manual_selection,
    formation_counts,
    legal_formations,
    selection_defaults,
    slot_positions,
)
from fpl_predictions.serving.services import (  # noqa: E402
    optimize_selection,
    rate_selection,
    sample_strong_selection,
)
from fpl_predictions.squads.schemas import SquadSelection  # noqa: E402
from fpl_predictions.squads.validation import (  # noqa: E402
    SquadValidationError,
    validate_squad,
)


BUNDLE_ROOT = Path(os.getenv("FPL_SERVING_BUNDLE", "deployment/current"))
SCORE_HELP = {
    "overall": (
        "Projected starting-XI points plus the extra captain points, compared "
        "with 1,000 realistic, legal squads at a similar budget."
    ),
    "goalkeeper": "How strongly the starting goalkeeper projects over this window.",
    "defence": "Combined projection from starting defenders.",
    "midfield": "Combined projection from starting midfielders.",
    "attack": "Combined projection from starting forwards.",
    "bench": "Combined projection of the four substitutes, shown separately.",
    "captaincy": "The selected captain's projected bonus contribution.",
}
POSITION_NAMES = {
    "GKP": "Goalkeeper",
    "DEF": "Defenders",
    "MID": "Midfielders",
    "FWD": "Forwards",
}
POSITION_SINGULAR = {
    "GKP": "Goalkeeper",
    "DEF": "Defender",
    "MID": "Midfielder",
    "FWD": "Forward",
}
CLUB_COLOURS = {
    "ARS": ("#ef0107", "#ffffff"),
    "AVL": ("#670e36", "#95bfe5"),
    "BOU": ("#da291c", "#ffffff"),
    "BRE": ("#e30613", "#ffffff"),
    "BHA": ("#0057b8", "#ffffff"),
    "CHE": ("#034694", "#ffffff"),
    "COV": ("#66ccff", "#10233f"),
    "CRY": ("#1b458f", "#ffffff"),
    "EVE": ("#003399", "#ffffff"),
    "FUL": ("#111111", "#ffffff"),
    "HUL": ("#f5a12d", "#111111"),
    "IPS": ("#3a64a3", "#ffffff"),
    "LEE": ("#ffcd00", "#1d428a"),
    "LIV": ("#c8102e", "#ffffff"),
    "MCI": ("#6cabdd", "#10233f"),
    "MUN": ("#da291c", "#ffffff"),
    "NEW": ("#171717", "#ffffff"),
    "NFO": ("#dd0000", "#ffffff"),
    "TOT": ("#132257", "#ffffff"),
    "SUN": ("#eb172b", "#ffffff"),
}


@st.cache_resource(show_spinner="Loading the latest predictions…")
def _bundle() -> Any:
    return load_serving_bundle(BUNDLE_ROOT)


@st.cache_data(show_spinner=False)
def _model_metadata(bundle_root: str, horizon: int) -> dict[str, Any]:
    path = Path(bundle_root) / "models" / "points" / f"horizon-{horizon}" / "metadata.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


@st.cache_data(ttl=3600, show_spinner=False)
def _portrait_available(url: str) -> bool:
    """Check whether the official portrait URL currently serves an image."""
    try:
        response = requests.head(
            url,
            allow_redirects=True,
            timeout=(0.8, 2.0),
            headers={"User-Agent": "FPL-Squad-Lab/1.0"},
        )
    except requests.RequestException:
        # A temporary server-side network issue should not suppress a portrait
        # that may still load normally in the user's browser.
        return True
    content_type = response.headers.get("content-type", "").lower()
    return response.status_code == 200 and content_type.startswith("image/")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --fpl-purple: #37003c;
          --fpl-purple-2: #5b1465;
          --fpl-green: #00ff87;
          --fpl-cyan: #04f5ff;
          --ink: #172033;
          --muted: #637083;
          --panel: rgba(255, 255, 255, .92);
        }
        .stApp {
          background:
            radial-gradient(circle at 90% 0%, rgba(4,245,255,.15), transparent 28rem),
            radial-gradient(circle at 0% 18%, rgba(0,255,135,.11), transparent 24rem),
            #f5f6fb;
          color: var(--ink);
        }
        [data-testid="stHeader"] { background: rgba(245,246,251,.82); }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #2a032f 0%, #43004a 100%);
        }
        [data-testid="stSidebar"] * { color: #fff; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div {
          background: #fff !important; border-color: rgba(255,255,255,.55) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] div,
        [data-testid="stSidebar"] [data-baseweb="select"] svg,
        [data-testid="stSidebar"] [data-baseweb="input"] input {
          color: #172033 !important; -webkit-text-fill-color: #172033 !important;
          fill: #172033 !important;
        }
        [data-baseweb="popover"], [data-baseweb="menu"] { background: #fff !important; }
        [data-baseweb="popover"] *, [data-baseweb="menu"] * {
          color: #172033 !important; -webkit-text-fill-color: #172033 !important;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * { color:#fff !important; }
        .block-container { max-width: 1240px; padding-top: 2.25rem; }
        h1, h2, h3 { letter-spacing: -.025em; }
        h1 { color: var(--fpl-purple); font-weight: 850 !important; }
        .hero-strip {
          position: relative; overflow: hidden; margin: .25rem 0 1.5rem;
          padding: 1.15rem 1.35rem; border-radius: 20px;
          background: linear-gradient(120deg, #37003c 0%, #6a1675 64%, #04aeb5 140%);
          color: white; box-shadow: 0 14px 36px rgba(55,0,60,.18);
        }
        .hero-strip:after {
          content: "⚽"; position: absolute; right: 1.1rem; top: -.9rem;
          font-size: 6rem; opacity: .12; transform: rotate(-14deg);
        }
        .hero-kicker { color: var(--fpl-green); font-weight: 800; font-size: .78rem;
          letter-spacing: .12em; text-transform: uppercase; }
        .hero-copy { max-width: 760px; margin-top: .3rem; font-size: 1.02rem; opacity: .94; }
        .hero-league-badge { display:inline-flex; align-items:center; gap:.45rem; margin-bottom:.55rem;
          padding:.28rem .62rem .28rem .32rem; border:1px solid rgba(255,255,255,.25);
          border-radius:999px; background:rgba(255,255,255,.1); color:#fff; font-size:.74rem;
          font-weight:750; letter-spacing:.02em; }
        .hero-league-badge span { display:grid; place-items:center; width:27px; height:27px;
          border-radius:50%; background:var(--fpl-green); color:#37003c; font-weight:950; }
        div[data-testid="stMetric"] {
          background: var(--panel); border: 1px solid rgba(55,0,60,.09);
          padding: .85rem 1rem; border-radius: 16px;
          box-shadow: 0 5px 18px rgba(39,31,48,.055);
        }
        div[data-testid="stFileUploader"], div[data-testid="stExpander"] {
          border-radius: 16px; overflow: hidden;
        }
        .stButton > button, .stDownloadButton > button {
          border-radius: 999px; font-weight: 750; min-height: 2.7rem;
        }
        .stButton > button[kind="primary"] {
          background: linear-gradient(90deg, #37003c, #681270);
          border: 0; box-shadow: 0 7px 18px rgba(55,0,60,.2); color:#fff !important;
        }
        .stButton > button[kind="primary"] * { color:#fff !important; }
        .stButton > button:disabled, .stButton > button[kind="primary"]:disabled {
          background:#e4e7ec !important; border:1px solid #c7ccd4 !important;
          color:#475467 !important; box-shadow:none !important; opacity:1 !important;
        }
        .stButton > button:disabled *, .stButton > button[kind="primary"]:disabled * {
          color:#475467 !important; -webkit-text-fill-color:#475467 !important;
        }
        .fpl-pitch {
          position: relative; overflow: visible; border-radius: 24px;
          padding: 2rem 1rem 1.4rem; margin: .8rem 0 1rem;
          background: repeating-linear-gradient(90deg,#079a51 0,#079a51 12.5%,#06934c 12.5%,#06934c 25%);
          box-shadow: inset 0 0 0 3px rgba(255,255,255,.23), 0 14px 35px rgba(4,82,45,.2);
        }
        .fpl-pitch:before {
          content:""; position:absolute; left:50%; top:0; bottom:0; width:2px;
          background:rgba(255,255,255,.48);
        }
        .fpl-pitch:after {
          content:""; position:absolute; width:130px; height:130px; border:2px solid rgba(255,255,255,.48);
          border-radius:50%; left:50%; top:50%; transform:translate(-50%,-50%);
        }
        .pitch-row { position:relative; z-index:1; display:flex; justify-content:space-evenly;
          align-items:end; gap:.45rem; margin: .6rem auto 1.05rem; }
        .player-card { position:relative; width: min(118px, 18vw); min-width:72px; text-align:center;
          filter:drop-shadow(0 5px 6px rgba(0,0,0,.18)); z-index:3; }
        .player-card[open], .player-card:hover, .player-card:focus-within { z-index:40; }
        .player-card summary { display:block; list-style:none; cursor:pointer; border-radius:10px;
          outline:none; -webkit-tap-highlight-color:transparent; }
        .player-card summary::-webkit-details-marker { display:none; }
        .player-card summary:focus-visible { box-shadow:0 0 0 3px #04f5ff; }
        .player-photo-wrap { position:relative; height:78px; display:flex; align-items:flex-end;
          justify-content:center; }
        .player-photo { position:absolute; inset:auto 0 0; margin:auto; height:78px; max-width:90px;
          object-fit:contain; object-position:center bottom; }
        .player-silhouette { width:62px; height:70px; display:flex; align-items:flex-end;
          justify-content:center; margin:0 auto; color:#d7dbe2; }
        .player-silhouette svg { width:58px; height:66px; filter:drop-shadow(0 3px 4px rgba(0,0,0,.18)); }
        .player-label { position:relative; border-radius:9px; overflow:hidden; background:white; }
        .club-band { height:5px; }
        .player-name { color:#172033; font-size:.78rem; font-weight:850; padding:.34rem .18rem .08rem;
          white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .player-meta { color:#637083; font-size:.61rem; font-weight:700; padding:0 .18rem .32rem; }
        .player-info-dot { display:inline-grid; place-items:center; width:13px; height:13px;
          margin-left:2px; border-radius:50%; background:#e9edf3; color:#37003c;
          font-size:.52rem; font-weight:950; vertical-align:1px; }
        .player-tooltip { position:absolute; left:50%; bottom:calc(100% + 9px); width:178px;
          transform:translate(-50%,6px); padding:.7rem; border:1px solid rgba(55,0,60,.12);
          border-radius:14px; background:#fff; color:#172033; text-align:left;
          box-shadow:0 14px 34px rgba(26,21,34,.24); opacity:0; visibility:hidden;
          pointer-events:none; transition:opacity .16s ease,transform .16s ease; z-index:100; }
        .player-card:hover .player-tooltip, .player-card[open] .player-tooltip,
        .player-card:focus-within .player-tooltip { opacity:1; visibility:visible;
          transform:translate(-50%,0); }
        .pitch-row > .player-card:first-child .player-tooltip { left:0; transform:translate(0,6px); }
        .pitch-row > .player-card:first-child:hover .player-tooltip,
        .pitch-row > .player-card:first-child[open] .player-tooltip { transform:translate(0,0); }
        .pitch-row > .player-card:last-child .player-tooltip { left:auto; right:0; transform:translate(0,6px); }
        .pitch-row > .player-card:last-child:hover .player-tooltip,
        .pitch-row > .player-card:last-child[open] .player-tooltip { transform:translate(0,0); }
        .tooltip-title { color:#37003c; font-size:.76rem; font-weight:900; margin-bottom:.45rem; }
        .tooltip-grid { display:grid; grid-template-columns:1fr 1fr; gap:.35rem; }
        .tooltip-stat { padding:.35rem; border-radius:8px; background:#f4f5f9; }
        .tooltip-value { display:block; color:#172033; font-size:.73rem; font-weight:900; }
        .tooltip-label { display:block; color:#667085; font-size:.52rem; font-weight:700;
          line-height:1.2; margin-top:.05rem; }
        .tooltip-hint { margin-top:.4rem; color:#737d8d; font-size:.52rem; text-align:center; }
        .captain-chip { position:absolute; top:-9px; left:-5px; width:22px; height:22px; border-radius:50%;
          display:grid; place-items:center; background:#37003c; color:white; border:2px solid #00ff87;
          font-size:.65rem; font-weight:900; z-index:2; }
        .bench-shell { background:linear-gradient(135deg,#d9f7ea,#d8f2f5); border-radius:18px;
          padding:.8rem; margin-top:.65rem; border:1px solid rgba(55,0,60,.08); }
        .bench-title { color:#37003c; font-size:.75rem; font-weight:900; letter-spacing:.09em;
          text-transform:uppercase; text-align:center; margin-bottom:.25rem; }
        .bench-shell .pitch-row { margin:.35rem 0 .2rem; }
        .sub-position-badge { margin:.1rem auto .35rem; width:max-content; padding:.18rem .48rem;
          border-radius:999px; background:rgba(55,0,60,.88); color:#fff; font-size:.61rem;
          font-weight:850; letter-spacing:.04em; text-transform:uppercase; }
        .blank-player { height:70px; width:58px; border:2px dashed rgba(255,255,255,.65);
          border-radius:50% 50% 12px 12px; display:grid; place-items:center; color:white;
          font-size:1.4rem; margin:0 auto 8px; }
        .st-key-manual_pitch, .st-key-edit_pitch {
          position:relative; border-radius:24px; padding:1.1rem 1rem 1.35rem;
          background:repeating-linear-gradient(90deg,#079a51 0,#079a51 12.5%,#06934c 12.5%,#06934c 25%);
          box-shadow:inset 0 0 0 3px rgba(255,255,255,.22),0 12px 30px rgba(4,82,45,.18);
        }
        .st-key-manual_pitch label, .st-key-edit_pitch label { color:white !important; font-weight:800; }
        .pitch-section-title { position:relative; z-index:2; text-align:center; color:white;
          font-size:.72rem; font-weight:900; letter-spacing:.1em; text-transform:uppercase;
          margin:.55rem 0 .2rem; text-shadow:0 1px 3px rgba(0,0,0,.2); }
        .guide-card { height:100%; background:white; border:1px solid rgba(55,0,60,.08);
          border-radius:18px; padding:1rem 1.05rem; box-shadow:0 7px 22px rgba(39,31,48,.055); }
        .guide-icon { font-size:1.45rem; }
        .guide-title { color:#37003c; font-weight:850; margin:.3rem 0 .35rem; }
        .guide-copy { color:#586579; font-size:.9rem; line-height:1.48; }
        .source-note { color:#6b7280; font-size:.75rem; text-align:center; margin-top:.3rem; }
        .creator-footer { margin-top:2.2rem; padding:1.1rem; border-top:1px solid rgba(55,0,60,.1);
          color:#687386; text-align:center; font-size:.82rem; }
        .creator-footer a { color:#37003c !important; font-weight:800; text-decoration:none; }
        .creator-footer a:hover { text-decoration:underline; }
        .sidebar-credit { margin-top:1.1rem; padding-top:.8rem; border-top:1px solid rgba(255,255,255,.17);
          color:rgba(255,255,255,.72); font-size:.72rem; line-height:1.55; }
        .sidebar-credit a { color:#fff !important; font-weight:800; text-decoration:none; }
        .sidebar-credit a:hover { text-decoration:underline; }
        .purpose-callout { margin:.25rem 0 1rem; padding:1rem 1.1rem; border-radius:16px;
          background:linear-gradient(120deg,rgba(55,0,60,.07),rgba(4,245,255,.1));
          border:1px solid rgba(55,0,60,.08); color:#334155; line-height:1.55; }
        @media(max-width:700px) {
          .block-container { padding:1.2rem .7rem; }
          .fpl-pitch { padding-left:.25rem; padding-right:.25rem; }
          .player-photo-wrap,.player-photo { height:58px; }
          .player-silhouette { height:54px; width:48px; }
          .player-silhouette svg { height:52px; width:46px; }
          .player-name { font-size:.64rem; }
          .player-meta { font-size:.53rem; }
          .player-tooltip { width:154px; }
          .pitch-row { gap:.15rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _set_selection(selection: SquadSelection, audit: dict[str, Any] | None = None) -> None:
    st.session_state["selection"] = selection.as_dict()
    st.session_state["recognition_audit"] = audit
    st.session_state.pop("rating_result", None)
    st.session_state.pop("optimization_result", None)
    # Streamlit otherwise preserves an old widget choice and ignores the new
    # formation's index when a screenshot or optimized squad replaces it.
    st.session_state.pop("edit_formation", None)


def _selection() -> SquadSelection | None:
    payload = st.session_state.get("selection")
    return SquadSelection.from_mapping(payload) if isinstance(payload, dict) else None


def _friendly_labels(players: pd.DataFrame) -> dict[int, str]:
    return {
        int(row.player_id): (
            f"{row.display_name} · {row.club_short_name} · £{float(row.price):.1f}m"
        )
        for row in players.itertuples()
    }


def _position_options(players: pd.DataFrame, position: str) -> list[int]:
    pool = players.loc[players["position_short_name"].eq(position)].copy()
    if "can_select" in pool:
        selectable = pool["can_select"].fillna(True).astype(bool)
        pool = pool.loc[selectable]
    return pool.sort_values(
        ["price", "display_name"], ascending=[False, True]
    )["player_id"].astype(int).tolist()


def _club_colours(short_name: str) -> tuple[str, str]:
    return CLUB_COLOURS.get(short_name, ("#37003c", "#ffffff"))


def _stat_value(value: Any, *, decimals: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def _friendly_timestamp(value: Any, *, include_time: bool = True) -> str:
    if not value:
        return "Unknown"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        local = parsed.astimezone(ZoneInfo("America/New_York"))
    except (TypeError, ValueError):
        return str(value)
    if not include_time:
        return local.strftime("%B %d, %Y").replace(" 0", " ")
    clock = local.strftime("%I:%M %p").lstrip("0")
    return f"{local.strftime('%B %d, %Y').replace(' 0', ' ')} at {clock} {local.tzname()}"


def _silhouette_html() -> str:
    return (
        '<div class="player-silhouette" aria-hidden="true">'
        '<svg viewBox="0 0 80 92" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="40" cy="23" r="16" fill="currentColor"/>'
        '<path d="M12 88c1-24 10-39 28-39s27 15 28 39H12Z" fill="currentColor"/>'
        '<path d="M24 54 40 65 56 54l7 11-10 23H27L17 65l7-11Z" fill="#b9c0ca"/>'
        '</svg></div>'
    )


def _player_card(
    row: pd.Series | None,
    marker: str = "",
    *,
    prediction: pd.Series | None = None,
    horizon: int = 3,
    snapshot_gameweek: int | None = None,
    position: str | None = None,
) -> str:
    if row is None:
        position_copy = POSITION_SINGULAR.get(position or "", "Player")
        return (
            '<div class="player-card"><div class="blank-player">+</div>'
            '<div class="player-label"><div class="club-band" style="background:#00ff87"></div>'
            f'<div class="player-name">Choose player</div><div class="player-meta">{position_copy} slot</div>'
            "</div></div>"
        )
    club = str(row.get("club_short_name") or "")
    accent, _ = _club_colours(club)
    name = escape(str(row["display_name"]))
    player_position = str(row.get("position_short_name") or position or "")
    photo = row.get("photo_url")
    marker_html = (
        f'<span class="captain-chip">{escape(marker)}</span>' if marker else ""
    )
    if pd.notna(photo) and photo and _portrait_available(str(photo)):
        image_html = (
            f'<img class="player-photo" src="{escape(str(photo), quote=True)}" '
            'alt="" loading="lazy">'
        )
    else:
        image_html = _silhouette_html()
    projected = None if prediction is None else prediction.get(f"predicted_points_{horizon}")
    start_probability = None if prediction is None else prediction.get("start_probability_1")
    if start_probability is not None and not pd.isna(start_probability):
        start_probability = 100 * float(start_probability)
    points_label = "Last season" if snapshot_gameweek == 1 else "Season points"
    tooltip_stats = [
        ("Points / game", _stat_value(row.get("points_per_game"))),
        (points_label, _stat_value(row.get("total_points"), decimals=0)),
        (f"Next {horizon} GW", _stat_value(projected)),
        ("Chance to start", _stat_value(start_probability, decimals=0, suffix="%")),
        ("Selected by", _stat_value(row.get("ownership_percent"), suffix="%")),
        ("Price", f"£{float(row['price']):.1f}m"),
    ]
    stats_html = "".join(
        '<div class="tooltip-stat">'
        f'<span class="tooltip-value">{escape(value)}</span>'
        f'<span class="tooltip-label">{escape(label)}</span></div>'
        for label, value in tooltip_stats
    )
    tooltip_html = (
        f'<div class="player-tooltip"><div class="tooltip-title">{name}</div>'
        f'<div class="tooltip-grid">{stats_html}</div>'
        '<div class="tooltip-hint">Model forecast · tap card again to close</div></div>'
    )
    return (
        f'<details class="player-card">{marker_html}<summary aria-label="View statistics for {name}">'
        f'<div class="player-photo-wrap">{image_html}</div>'
        f'<div class="player-label"><div class="club-band" style="background:{accent}"></div>'
        f'<div class="player-name">{name}<span class="player-info-dot">i</span></div>'
        f'<div class="player-meta">{escape(club)} · {escape(player_position)} · £{float(row["price"]):.1f}m</div>'
        f"</div></summary>{tooltip_html}</details>"
    )


def _pitch_html(bundle: Any, selection: SquadSelection, horizon: int) -> str:
    indexed = bundle.players.drop_duplicates("player_id").set_index("player_id")
    predictions = bundle.predictions.drop_duplicates("player_id").set_index("player_id")
    snapshot_gameweek = bundle.manifest.get("snapshot_gameweek")
    rows: list[str] = []
    for position in POSITION_ORDER:
        player_ids = [
            player_id
            for player_id in selection.starting_xi
            if str(indexed.loc[player_id, "position_short_name"]) == position
        ]
        cards = []
        for player_id in player_ids:
            marker = (
                "C" if player_id == selection.captain else "V" if player_id == selection.vice_captain else ""
            )
            cards.append(
                _player_card(
                    indexed.loc[player_id],
                    marker,
                    prediction=predictions.loc[player_id],
                    horizon=horizon,
                    snapshot_gameweek=snapshot_gameweek,
                )
            )
        rows.append(f'<div class="pitch-row">{"".join(cards)}</div>')
    bench_cards = []
    for player_id in selection.bench:
        bench_cards.append(
            _player_card(
                indexed.loc[player_id],
                prediction=predictions.loc[player_id],
                horizon=horizon,
                snapshot_gameweek=snapshot_gameweek,
            )
        )
    return (
        f'<div class="fpl-pitch">{"".join(rows)}</div>'
        '<div class="bench-shell"><div class="bench-title">Substitutes</div>'
        f'<div class="pitch-row">{"".join(bench_cards)}</div></div>'
        '<div class="source-note">Player portraits are supplied by the official Premier League data feed.</div>'
    )


def _player_table(bundle: Any, selection: SquadSelection, horizon: int) -> pd.DataFrame:
    indexed = bundle.players.set_index("player_id")
    predictions = bundle.predictions.set_index("player_id")
    records = []
    for role, values in (("Starter", selection.starting_xi), ("Sub", selection.bench)):
        for order, player_id in enumerate(values, start=1):
            row = indexed.loc[player_id]
            prediction = predictions.loc[player_id]
            records.append(
                {
                    "Role": role,
                    "Order": order,
                    "Player": row["display_name"],
                    "Club": row["club_short_name"],
                    "Pos": row["position_short_name"],
                    "Price": f"£{float(row['price']):.1f}m",
                    "Captain": "C" if player_id == selection.captain else "V" if player_id == selection.vice_captain else "",
                    f"Projected ({horizon} GW)": round(
                        float(prediction.get(f"predicted_points_{horizon}", 0.0)),
                        2,
                    ),
                }
            )
    return pd.DataFrame(records)


def _recognition_table(audit: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Slot": row["slot_id"],
            "Text read": " / ".join(row["ocr_readings"]),
            "Matched player": row["display_name"],
            "Confidence": f"{100 * row['match_score']:.1f}%",
            "Other possibilities": ", ".join(
                item["display_name"] for item in row["alternatives"][1:3]
            ),
        }
        for row in audit["slots"]
    )


def _render_slot(
    bundle: Any,
    position: str,
    default: int,
    *,
    key: str,
    label: str,
    horizon: int,
) -> int:
    labels = _friendly_labels(bundle.players)
    options = [0] + _position_options(bundle.players, position)
    if default not in options:
        default = 0
    holder = st.empty()
    chosen = st.selectbox(
        label,
        options,
        index=options.index(default),
        format_func=lambda value: (
            f"Choose {POSITION_NAMES[position].lower().rstrip('s')}…"
            if value == 0
            else labels[value]
        ),
        key=key,
        help=f"Select one current FPL {POSITION_NAMES[position].lower().rstrip('s')}.",
        label_visibility="collapsed",
    )
    row = None
    if int(chosen) > 0:
        row = bundle.players.set_index("player_id").loc[int(chosen)]
    prediction = None
    if int(chosen) > 0:
        prediction = bundle.predictions.set_index("player_id").loc[int(chosen)]
    holder.markdown(
        _player_card(
            row,
            prediction=prediction,
            horizon=horizon,
            snapshot_gameweek=bundle.manifest.get("snapshot_gameweek"),
            position=position,
        ),
        unsafe_allow_html=True,
    )
    return int(chosen)


def _render_interactive_pitch(
    bundle: Any,
    formation: str,
    existing: SquadSelection | None,
    *,
    key_prefix: str,
    submit_label: str,
    source: str,
    budget_limit: float | None,
    horizon: int,
) -> None:
    starter_positions, bench_positions = slot_positions(formation, bundle.rules)
    starter_defaults, bench_defaults = selection_defaults(
        existing, formation, bundle.players, bundle.rules
    )
    selected_starters: list[int] = []
    selected_bench: list[int] = []
    with st.container(key=f"{key_prefix}_pitch"):
        offset = 0
        counts = formation_counts(formation, bundle.rules)
        for position in POSITION_ORDER:
            count = counts[position]
            st.markdown(
                f'<div class="pitch-section-title">{POSITION_NAMES[position]}</div>',
                unsafe_allow_html=True,
            )
            columns = st.columns(count)
            for index in range(count):
                absolute = offset + index
                with columns[index]:
                    selected_starters.append(
                        _render_slot(
                            bundle,
                            position,
                            starter_defaults[absolute],
                            key=f"{key_prefix}_{formation}_starter_{absolute}",
                            label=f"Starter {absolute + 1} · {position}",
                            horizon=horizon,
                        )
                    )
            offset += count
        st.markdown(
            '<div class="pitch-section-title">Substitutes</div>',
            unsafe_allow_html=True,
        )
        columns = st.columns(4)
        for index, position in enumerate(bench_positions):
            with columns[index]:
                st.markdown(
                    f'<div class="sub-position-badge">{POSITION_SINGULAR[position]}</div>',
                    unsafe_allow_html=True,
                )
                selected_bench.append(
                    _render_slot(
                        bundle,
                        position,
                        bench_defaults[index],
                        key=f"{key_prefix}_{formation}_bench_{index}",
                        label=f"Substitute {index + 1} · {position}",
                        horizon=horizon,
                    )
                )

    chosen_starters = [value for value in selected_starters if value > 0]
    captain_default = existing.captain if existing and existing.captain in chosen_starters else 0
    vice_default = existing.vice_captain if existing and existing.vice_captain in chosen_starters else 0
    labels = _friendly_labels(bundle.players)
    captain_options = [0] + list(dict.fromkeys(chosen_starters))
    choice_cols = st.columns(3)
    captain = choice_cols[0].selectbox(
        "Captain",
        captain_options,
        index=captain_options.index(captain_default),
        format_func=lambda value: "Choose captain…" if value == 0 else labels[value],
        key=f"{key_prefix}_{formation}_captain",
        help="Captain must be in the starting XI and receives double projected points.",
    )
    vice = choice_cols[1].selectbox(
        "Vice-captain",
        captain_options,
        index=captain_options.index(vice_default),
        format_func=lambda value: "Choose vice-captain…" if value == 0 else labels[value],
        key=f"{key_prefix}_{formation}_vice",
        help="Vice-captain takes over if the captain does not play.",
    )
    bank_default = float(existing.bank) if existing and existing.bank is not None else 0.0
    bank = choice_cols[2].number_input(
        "Money in bank (£m)",
        min_value=0.0,
        max_value=20.0,
        value=bank_default,
        step=0.1,
        key=f"{key_prefix}_{formation}_bank",
        help="Optional cash remaining for future transfers; use 0 if unknown.",
    )

    all_players = selected_starters + selected_bench
    complete = all(value > 0 for value in all_players)
    unique = len(set(all_players)) == len(all_players) if complete else False
    indexed = bundle.players.set_index("player_id")
    cost = sum(float(indexed.loc[value, "price"]) for value in all_players if value > 0)
    status_cols = st.columns(3)
    status_cols[0].metric("Formation", formation, help="The shape of your starting XI.", border=True)
    status_cols[1].metric("Players selected", f"{sum(value > 0 for value in all_players)}/15", border=True)
    status_cols[2].metric(
        "Current cost",
        f"£{cost:.1f}m",
        help="£100m is the maximum allowed squad value.",
        border=True,
    )
    if complete and not unique:
        st.warning("A player has been selected more than once. Every slot must be unique.")
    if complete and cost > bundle.rules.budget + 1e-9:
        st.warning(
            f"This squad costs £{cost:.1f}m. The maximum is £{bundle.rules.budget:.1f}m. "
            "You can save it and keep editing, but it cannot be rated or optimized yet."
        )
    submitted = st.button(
        submit_label,
        type="primary",
        width="stretch",
        disabled=(
            not complete
            or not unique
            or int(captain) <= 0
            or int(vice) <= 0
            or int(captain) == int(vice)
        ),
        key=f"{key_prefix}_{formation}_submit",
    )
    if submitted:
        try:
            candidate = build_manual_selection(
                selected_starters,
                selected_bench,
                int(captain),
                int(vice),
                bank=float(bank),
                budget_limit=budget_limit,
                source=source,
            )
            validated = validate_squad(
                candidate,
                bundle.players,
                bundle.rules,
                enforce_budget=False,
            )
        except (SquadValidationError, ValueError) as exc:
            st.error(str(exc))
        else:
            _set_selection(candidate)
            if validated.total_cost > bundle.rules.budget + 1e-9:
                st.warning(
                    f"Squad saved at £{validated.total_cost:.1f}m. The FPL maximum is "
                    f"£{bundle.rules.budget:.1f}m, so you will need to reduce the cost "
                    "before rating or improving it."
                )
            else:
                st.success(
                    f"Squad saved: {validated.formation}, £{validated.total_cost:.1f}m."
                )
            st.rerun()


def _render_screenshot_input(bundle: Any) -> None:
    upload_col, action_col = st.columns([2, 1])
    with upload_col:
        uploaded = st.file_uploader(
            "Upload an FPL Pick Team screenshot",
            type=["png", "jpg", "jpeg"],
            help=(
                "The recognizer locates 11 starters and four substitutes, reads "
                "their nameplates locally with Tesseract, and checks the result "
                "against current FPL squad rules. Images are not retained."
            ),
        )
    with action_col:
        st.write("")
        st.write("")
        recognize_clicked = st.button(
            "Read my screenshot",
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
            with st.spinner("Finding the pitch and reading player names…"):
                recognition = recognize_screenshot(
                    temporary_path,
                    bundle.players,
                    bundle.rules,
                )
            _set_selection(recognition.selection, recognition.audit)
            st.success(
                f"Found a {recognition.audit['formation']} squad · "
                f"average name confidence {100 * recognition.audit['mean_match_score']:.1f}%"
            )
            st.rerun()
        except (ScreenshotRecognitionError, OSError, ValueError) as exc:
            st.error(str(exc))
            st.info("If this image remains unreadable, choose Build manually and enter the squad directly.")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _guide_card(icon: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="guide-card"><div class="guide-icon">{escape(icon)}</div>'
        f'<div class="guide-title">{escape(title)}</div>'
        f'<div class="guide-copy">{escape(copy)}</div></div>',
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="FPL Squad Lab",
    page_icon="⚽",
    layout="wide",
)
_inject_styles()
st.title("FPL Squad Lab")
st.markdown(
    """
    <div class="hero-strip">
      <div class="hero-league-badge"><span>PL</span> Premier League fantasy analysis</div>
      <div class="hero-kicker">Your FPL squad, explained</div>
      <div class="hero-copy">Upload a Fantasy Premier League screenshot or pick the players yourself. You can see your squad on the pitch, check its 0-100 rating, and find transfers that may improve it.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    bundle = _bundle()
except (ServingBundleError, FileNotFoundError, OSError, ValueError) as exc:
    st.error(f"The current FPL data could not be loaded: {exc}")
    st.stop()

manifest = bundle.manifest
with st.sidebar:
    st.header("Prediction settings")
    horizon = st.selectbox(
        "How far ahead?",
        bundle.horizons,
        index=(bundle.horizons.index(3) if 3 in bundle.horizons else 0),
        format_func=lambda value: f"Next {value} Gameweek{'s' if value != 1 else ''}",
        help=(
            "One Gameweek emphasizes the immediate fixture. Three or five "
            "Gameweeks reward players with a stronger short-term run."
        ),
    )
    st.divider()
    st.subheader("Latest model update", help="The data snapshot used by every rating and recommendation.")
    st.write(f"**Season:** {manifest.get('season') or 'Unknown'}")
    st.write(
        f"**Player data:** Gameweek {manifest.get('snapshot_gameweek') or 'Unknown'} snapshot · "
        f"{_friendly_timestamp(manifest.get('snapshot_timestamp'), include_time=False)}"
    )
    st.write(
        f"**Predictions refreshed:** "
        f"{_friendly_timestamp(manifest.get('prediction_created_at_utc'))}"
    )
    st.caption("This tells you exactly how current the player data and forecasts are. Ratings and transfer suggestions use the same update.")
    st.markdown(
        """
        <div class="sidebar-credit">Created by <strong>Max Homm</strong>, a Computer Science student at the University of Waterloo.<br>
        <a href="https://mchomm.github.io/" target="_blank" rel="noopener noreferrer">Portfolio</a> ·
        <a href="https://www.linkedin.com/in/max-homm/" target="_blank" rel="noopener noreferrer">LinkedIn</a></div>
        """,
        unsafe_allow_html=True,
    )

selection = _selection()
if selection is None:
    st.subheader(
        "Add your squad",
        help="Screenshot recognition is optional. Manual entry supports every legal FPL formation.",
    )
    entry_method = st.segmented_control(
        "Choose an input method",
        ["📸 Screenshot", "✍️ Build manually"],
        default="📸 Screenshot",
        help="Both routes create the same editable 15-player squad.",
    )
    if entry_method == "📸 Screenshot":
        _render_screenshot_input(bundle)
    else:
        formations = legal_formations(bundle.rules)
        formation = st.selectbox(
            "Starting formation",
            formations,
            index=formations.index("3-4-3") if "3-4-3" in formations else 0,
            help="All formations shown here satisfy the official minimum and maximum starters by position.",
        )
        _render_interactive_pitch(
            bundle,
            formation,
            None,
            key_prefix="manual",
            submit_label="Save manual squad",
            source="manual",
            budget_limit=bundle.rules.budget,
            horizon=horizon,
        )
    st.markdown("#### Or let the model start for you")
    st.caption(
        "Choose the exact highest-projected squad, or generate a different strong "
        "legal squad each time. You can edit either one afterwards."
    )
    generator_cols = st.columns(2)
    exact_generated = generator_cols[0].button(
        "Generate best possible squad",
        width="stretch",
    )
    varied_generated = generator_cols[1].button(
        "Generate a strong varied squad",
        width="stretch",
        help=(
            "Adds a small random variation to the player forecasts before solving "
            "the same legal squad problem. Displayed points use the original forecasts."
        ),
    )
    if exact_generated or varied_generated:
        with st.spinner("Building a strong legal squad…"):
            optimized = (
                sample_strong_selection(bundle, horizon)
                if varied_generated
                else optimize_selection(bundle, horizon)
            )
        _set_selection(optimized.selection)
        generated_payload = optimized.as_dict(
            bundle.players, bundle.predictions
        )
        generated_payload["generation_mode"] = (
            "Strong varied squad" if varied_generated else "Exact best squad"
        )
        st.session_state["optimization_result"] = generated_payload
        st.rerun()
    with st.expander("New here? What this app does"):
        st.markdown(
            """
            1. **Add your 15-player FPL squad** from a screenshot, manually, or with the optimizer.
            2. **Choose one, three, or five Gameweeks** in the sidebar.
            3. **Rate the squad** to compare its forecast with realistic, legal squads at a similar budget.
            4. **Explore improvements** that respect Premier League fantasy prices, positions, club limits, and transfer costs.

            This app is meant to make Premier League and FPL data easier to use. Forecasts are estimates, not guarantees.
            """
        )
    st.stop()

with st.expander("Replace this squad", expanded=False):
    replacement = st.segmented_control(
        "Choose another input method",
        ["📸 Screenshot", "✍️ Build manually"],
        default=None,
        key="replacement_method",
    )
    if replacement == "📸 Screenshot":
        _render_screenshot_input(bundle)
    elif replacement == "✍️ Build manually":
        replacement_formations = legal_formations(bundle.rules)
        replacement_formation = st.selectbox(
            "New formation",
            replacement_formations,
            index=(
                replacement_formations.index("3-4-3")
                if "3-4-3" in replacement_formations
                else 0
            ),
            key="replacement_formation",
            help="You can choose any legal FPL formation.",
        )
        _render_interactive_pitch(
            bundle,
            replacement_formation,
            None,
            key_prefix="manual",
            submit_label="Replace with manual squad",
            source="manual",
            budget_limit=bundle.rules.budget,
            horizon=horizon,
        )

audit = st.session_state.get("recognition_audit")
if isinstance(audit, dict):
    if audit.get("review_required"):
        st.warning(
            "A few names need checking: " + ", ".join(audit["low_confidence_slots"])
        )
    with st.expander(
        "Screenshot reading details",
        expanded=bool(audit.get("review_required")),
    ):
        st.caption(
            "Confidence measures how closely the OCR text matched a current player name. It is not a prediction of player performance."
        )
        st.dataframe(_recognition_table(audit), hide_index=True, width="stretch")

try:
    display_validated = validate_squad(
        selection,
        bundle.players,
        bundle.rules,
        enforce_budget=False,
    )
except SquadValidationError as exc:
    st.error(str(exc))
    st.stop()

maximum_budget = float(bundle.rules.budget)
squad_over_budget = display_validated.total_cost > maximum_budget + 1e-9
if squad_over_budget:
    st.warning(
        f"This squad is worth £{display_validated.total_cost:.1f}m, but the FPL maximum "
        f"is £{maximum_budget:.1f}m. You can still view and edit it. Reduce the squad "
        "value before using the rating or transfer tools."
    )

start_tab, current_tab, rating_tab, improve_tab, model_tab = st.tabs(
    ["👋 Start here", "⚽ My squad", "📊 Squad rating", "↗ Improve my team", "? Model guide"],
    default="⚽ My squad",
)

with start_tab:
    st.subheader("A clearer way to understand your Premier League fantasy squad")
    st.markdown(
        """
        <div class="purpose-callout">
          FPL Squad Lab helps answer three practical questions: <strong>How good is my squad? Where is it strongest? Which transfers might help?</strong> It uses official FPL player information and historical results, and every player remains editable after a screenshot is read.
        </div>
        """,
        unsafe_allow_html=True,
    )
    guide_columns = st.columns(4)
    with guide_columns[0]:
        _guide_card("1️⃣", "Add all 15 players", "Upload a Pick Team screenshot, enter every player manually, or let the optimizer build a legal starting point.")
    with guide_columns[1]:
        _guide_card("2️⃣", "Choose your window", "Use the sidebar to focus on the next one, three, or five Premier League Gameweeks.")
    with guide_columns[2]:
        _guide_card("3️⃣", "Rate the squad", "See projected points and simple 0-100 ratings for the full squad and each part of the team.")
    with guide_columns[3]:
        _guide_card("4️⃣", "Explore improvements", "Ask the optimizer for legal transfers, accounting for prices, club limits, formation, and transfer hits.")
    st.info("Tip: hover over a player on desktop, or tap the card on mobile, to see form, ownership, starting likelihood, and the current forecast.")
    st.caption("Forecasts are estimates, not guarantees. Late team news and real football will always create uncertainty.")

with current_tab:
    try:
        current_validated = display_validated
        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Formation",
            current_validated.formation,
            help="The current starting shape, inferred from the selected players.",
            border=True,
        )
        metric_cols[1].metric(
            "Squad value",
            f"£{current_validated.total_cost:.1f}m",
            help="Sum of the current FPL prices for all 15 players.",
            border=True,
        )
        metric_cols[2].metric(
            "Money in bank",
            "Unknown" if selection.bank is None else f"£{selection.bank:.1f}m",
            help="Cash available for transfers. Screenshot imports cannot determine this automatically.",
            border=True,
        )
        st.markdown(_pitch_html(bundle, selection, horizon), unsafe_allow_html=True)
    except SquadValidationError as exc:
        st.error(str(exc))

    with st.expander("Player list and projections"):
        st.dataframe(
            _player_table(bundle, selection, horizon),
            hide_index=True,
            width="stretch",
        )
    with st.expander("Edit players or change formation"):
        edit_formations = legal_formations(bundle.rules)
        current_formation = current_validated.formation
        edit_formation = st.selectbox(
            "Formation",
            edit_formations,
            index=edit_formations.index(current_formation),
            key="edit_formation",
            help="Changing shape moves players between the XI and bench where possible; review every slot before applying.",
        )
        _render_interactive_pitch(
            bundle,
            edit_formation,
            selection,
            key_prefix="edit",
            submit_label="Apply squad changes",
            source=selection.source,
            budget_limit=selection.budget_limit,
            horizon=horizon,
        )

with rating_tab:
    st.subheader(
        "How strong is this squad?",
        help="Ratings compare your forecast with 1,000 realistic, legal squads at a similar budget.",
    )
    st.caption("A score around 75 is typical; 80 is good, 90+ is excellent, and 95 is exceptional.")
    if squad_over_budget:
        st.warning(
            f"Rating is unavailable while the squad is above the £{maximum_budget:.1f}m limit."
        )
    if st.button(
        "Rate this squad",
        type="primary",
        width="stretch",
        disabled=squad_over_budget,
    ):
        try:
            with st.spinner("Projecting points and comparing similar legal squads…"):
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
        ordered = ["overall", "goalkeeper", "defence", "midfield", "attack", "bench", "captaincy"]
        first = st.columns(4)
        second = st.columns(3)
        for column, name in zip(first + second, ordered, strict=True):
            column.metric(
                name.title(),
                f"{scores[name]:.1f}",
                help=SCORE_HELP[name],
                border=True,
            )
        projection = rating_result["projection"]
        st.info(
            f"The model projects **{projection['overall_points']:.2f} points** "
            f"across the next {horizon} Gameweek{'s' if horizon != 1 else ''}, including captaincy."
        )
        strengths = rating_result["rating"].get("strengths", [])
        weaknesses = rating_result["rating"].get("weaknesses", [])
        insight_cols = st.columns(2)
        insight_cols[0].success(
            "Strong areas: " + (", ".join(name.title() for name in strengths) if strengths else "No standout area yet")
        )
        insight_cols[1].warning(
            "Areas to review: " + (", ".join(name.title() for name in weaknesses) if weaknesses else "No major weakness detected")
        )
        with st.expander("Player-by-player projection"):
            st.dataframe(pd.DataFrame(rating_result["players"]), hide_index=True, width="stretch")
        with st.expander("Rating calculation details"):
            st.write(rating_result["rating"]["interpretation"])
            st.caption(
                "The 0-100 score is a presentation scale. Raw projected points and percentile rank preserve the underlying model ordering."
            )
        st.download_button(
            "Download rating JSON",
            json.dumps(rating_result, indent=2),
            file_name="fpl-rating.json",
            mime="application/json",
        )

with improve_tab:
    st.subheader(
        "Find better moves",
        help="The optimizer searches legal squads exactly; it does not simply swap in the highest-scoring individual player.",
    )
    if squad_over_budget:
        st.warning(
            f"Transfer suggestions are unavailable while the squad is above the £{maximum_budget:.1f}m limit."
        )
    control_cols = st.columns(3)
    max_transfers = control_cols[0].slider(
        "Maximum transfers",
        1,
        5,
        3,
        help="Limits how many current players can be replaced.",
    )
    preseason = control_cols[1].checkbox(
        "Preseason / transfers are free",
        value=bool(manifest.get("snapshot_gameweek") == 1),
        help="When enabled, transfer hits are not deducted.",
    )
    free_transfers = control_cols[2].number_input(
        "Available free transfers",
        min_value=0,
        max_value=5,
        value=max_transfers if preseason else 1,
        disabled=preseason,
        help="Extra transfers beyond this number cost four projected points each.",
    )
    improve_col, rebuild_col, varied_col = st.columns(3)
    improve = improve_col.button(
        "Suggest transfers",
        type="primary",
        width="stretch",
        disabled=squad_over_budget,
    )
    rebuild = rebuild_col.button(
        "Generate best squad from scratch",
        width="stretch",
        disabled=squad_over_budget,
    )
    varied = varied_col.button(
        "Generate strong varied squad",
        width="stretch",
        disabled=squad_over_budget,
        help=(
            "Creates a different high-scoring legal squad on each click using a "
            "small random variation around the model forecasts."
        ),
    )
    if improve or rebuild or varied:
        try:
            with st.spinner("Searching all legal combinations…"):
                optimized = (
                    sample_strong_selection(bundle, horizon)
                    if varied
                    else optimize_selection(
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
                )
            optimization_payload = optimized.as_dict(
                bundle.players, bundle.predictions
            )
            optimization_payload["generation_mode"] = (
                "Strong varied squad"
                if varied
                else "Exact best squad"
                if rebuild
                else "Transfer suggestions"
            )
            st.session_state["optimization_result"] = optimization_payload
        except (RuntimeError, SquadValidationError, ValueError) as exc:
            st.error(str(exc))
    optimization = st.session_state.get("optimization_result")
    if isinstance(optimization, dict):
        if optimization.get("generation_mode"):
            st.caption(f"Result type: {optimization['generation_mode']}")
        projection = optimization["projection"]
        metric_cols = st.columns(4)
        metric_cols[0].metric("Formation", projection["formation"], border=True)
        metric_cols[1].metric(
            "Projected points",
            f"{projection['overall_points']:.2f}",
            help=SCORE_HELP["overall"],
            border=True,
        )
        metric_cols[2].metric(
            "Gain before hits",
            "New squad" if optimization["projected_points_gain"] is None else f"{optimization['projected_points_gain']:+.2f}",
            help="Projected improvement before transfer-point deductions.",
            border=True,
        )
        metric_cols[3].metric(
            "Gain after hits",
            "New squad" if optimization["net_projected_points_gain"] is None else f"{optimization['net_projected_points_gain']:+.2f}",
            help="Projected improvement after any four-point transfer hits.",
            border=True,
        )
        outgoing = optimization["transfers_out"]
        incoming = optimization["transfers_in"]
        if incoming:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Sell": [row["display_name"] for row in outgoing],
                        "Buy": [row["display_name"] for row in incoming],
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        optimized_selection = SquadSelection.from_mapping(optimization["selection"])
        st.markdown(_pitch_html(bundle, optimized_selection, horizon), unsafe_allow_html=True)
        with st.expander("Full optimized player list"):
            st.dataframe(pd.DataFrame(optimization["selected_players"]), hide_index=True, width="stretch")
        if st.button("Use this optimized squad", type="primary"):
            _set_selection(optimized_selection)
            st.rerun()
        st.download_button(
            "Download optimization JSON",
            json.dumps(optimization, indent=2),
            file_name="fpl-optimization.json",
            mime="application/json",
        )

with model_tab:
    st.subheader("How the predictions and ratings work")
    guide_columns = st.columns(3)
    with guide_columns[0]:
        _guide_card(
            "🧠",
            "Player forecasts",
            "A random-forest model learns recurring relationships between historical FPL performance, playing time, form, expected statistics, team strength, and upcoming opponents.",
        )
    with guide_columns[1]:
        _guide_card(
            "⏱️",
            "Line-up likelihood",
            "Separate models estimate appearances, starts, and minutes. Club-level reconciliation prevents a team from being treated as if too many players can start at once.",
        )
    with guide_columns[2]:
        _guide_card(
            "📏",
            "Squad rating",
            "Your projected score is compared with 1,000 legal, budget-matched squads. The result is shown on a familiar scale where the middle reference squad scores 75.",
        )

    metadata = _model_metadata(str(bundle.root), horizon)
    evaluation = metadata.get("evaluation", {})
    leaderboard = evaluation.get("leaderboard", [])
    selected_name = metadata.get("selected_model", "unknown model")
    selected_metrics = next(
        (row for row in leaderboard if row.get("model") == selected_name),
        {},
    )
    st.markdown("### Training snapshot")
    training_cols = st.columns(4)
    training_cols[0].metric(
        "Training examples",
        f"{int(metadata.get('training_rows', 0)):,}" if metadata else "Unknown",
        help="Each example is one player observed at a historical Gameweek deadline.",
        border=True,
    )
    training_cols[1].metric(
        "Model type",
        str(selected_name).replace("_", " ").title(),
        help="Random forests combine many decision trees and average their forecasts.",
        border=True,
    )
    training_cols[2].metric(
        "Validation MAE",
        f"{float(selected_metrics['mae']):.2f}" if "mae" in selected_metrics else "Unknown",
        help="Mean absolute error: the average absolute difference between prediction and outcome in historical time-based tests.",
        border=True,
    )
    training_cols[3].metric(
        "Ranking correlation",
        f"{float(selected_metrics['ranking_correlation']):.2f}" if selected_metrics.get("ranking_correlation") is not None else "Unknown",
        help="How well the model ordered players from weaker to stronger in historical tests; closer to 1 is better.",
        border=True,
    )

    with st.expander("What information does the model use?"):
        features = metadata.get("feature_schema", {})
        st.markdown(
            """
            - **FPL output:** points, minutes, starts, goals, assists, clean sheets, bonus and BPS.
            - **Underlying performance:** expected goals, expected assists, influence, creativity, threat and ICT index.
            - **Recent evidence:** recent points, minutes and how many recent Gameweeks are available.
            - **Fixtures and teams:** home/away fixture counts, opponent attack/defence strength, and the player's club strength.
            - **Player context:** position, club and current price.

            Missing inputs are handled by the trained preprocessing pipeline. Current injury/availability percentages are applied to the immediate forecast after the base model prediction.
            """
        )
        st.caption(
            f"This horizon uses {len(features.get('numeric', []))} numeric and {len(features.get('categorical', []))} categorical feature fields."
        )
    with st.expander("What do the scores mean?"):
        st.markdown(
            """
            - **Projected points** are the raw model forecast for the selected window.
            - **Percentile** is the percentage of reference squads projected below yours.
            - **0-100 rating** is a school-style presentation of that rank: roughly 75 is typical, 80 is good, 90+ is excellent and 95 is exceptional.
            - **Overall** uses the starting XI plus the extra captain contribution; it is not the average of the position ratings.
            - **Bench** is reported separately because substitutes only contribute through legal automatic substitutions.
            """
        )
    with st.expander("Technical version details"):
        st.json(
            {
                "season": manifest.get("season"),
                "data_snapshot_gameweek": manifest.get("snapshot_gameweek"),
                "data_snapshot_timestamp": manifest.get("snapshot_timestamp"),
                "prediction_created_at_utc": manifest.get("prediction_created_at_utc"),
                "model_runs": manifest.get("model_runs"),
                "prediction_horizons": manifest.get("horizons"),
                "reference_squads": manifest.get("reference_configuration"),
                "training_coverage": metadata.get("training_coverage"),
                "validation_folds": evaluation.get("folds"),
            }
        )
        st.caption(
            "Uploaded screenshots are processed temporarily and are not retained. The deployed model files are checksummed so the displayed version can be audited."
        )

st.markdown(
    """
    <div class="creator-footer">
      Built by <strong>Max Homm</strong>, a Computer Science student at the University of Waterloo.<br>
      <a href="https://mchomm.github.io/" target="_blank" rel="noopener noreferrer">Portfolio</a> ·
      <a href="https://www.linkedin.com/in/max-homm/" target="_blank" rel="noopener noreferrer">LinkedIn</a><br>
      This is an independent project and is not affiliated with the Premier League.
    </div>
    """,
    unsafe_allow_html=True,
)
