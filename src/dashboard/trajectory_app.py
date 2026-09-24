"""
Multi-Camera Vehicle Trajectory Reconstruction System -- Streamlit dashboard.

Run with:
    streamlit run src/dashboard/trajectory_app.py

This is the multi-camera console. The original single-gate dashboard is
untouched and still runs (`streamlit run src/dashboard/app.py`); the two
read the same database and neither depends on the other.

Layout
    HEADER           title + live session state, and the blacklist alert
    HOME tab         blacklisted vehicles, camera feeds, search by plate
                     number, search by description, traffic heatmap
    OPERATIONS tab   LEFT   per-camera source + START PROCESSING
                     CENTER live processing status (camera, progress, plate,
                            vehicle, FPS, logs)
                     RIGHT  statistics, including detections per camera
    TRAJECTORY tab   search result: vehicle, summary, timeline, map,
                     history table

Home is what an operator leaves open; Operations holds the controls they
touch when starting a run. Either search on Home opens the Trajectory tab.

Processing runs on a background thread owned by the CameraManager singleton
(see cameras/manager.py). Streamlit re-runs this script top to bottom on
every interaction, so nothing here may own long-lived state: the manager
lives at module scope in its own module, and progress is read back from the
shared status file.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path
from typing import Optional

# Add project root to path, so `streamlit run src/dashboard/trajectory_app.py`
# resolves `src.*` the same way `python -m` would.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import streamlit.components.v1 as components

from src.cameras.manager import get_camera_manager, reset_camera_manager
from src.cameras.preview_server import get_preview_server
from src.cameras.models import CameraState, SessionState
from src.cameras.registry import CameraConfigError
from src.cameras.sources import VideoSourceError
from src.cameras.stream_probe import grab_preview_frame, mask_credentials, probe_stream
from src.database import db as database
from src.database.vehicle_profiles import get_profile, search_profiles
from src.search.nl_query import parse_query
from src.alerts.blacklist import get_blacklist
from src.mapping import render_trajectory_map
from src.mapping.heatmap import render_traffic_heatmap as render_traffic_heatmap_html
from src.trajectory import TrajectoryEngine, trajectory_to_geojson
from src.utils.config import load_config
from src.utils.data_reset import delete_processing_session

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
DEFAULT_CAMERA_CONFIG_PATH = REPO_ROOT / "config" / "camera_config.yaml"


def camera_config_path() -> Path:
    """Where this page reads its cameras from.

    ALPR_CAMERA_CONFIG mirrors ALPR_DB_PATH: it points the page at another
    camera registry without editing the shipped one, so a test (or a second
    deployment on the same machine) can work on its own file. Read on every
    call rather than frozen at import, because this module stays in
    sys.modules across reruns -- a value captured once would outlive the
    environment it was read from.
    """
    return Path(os.getenv("ALPR_CAMERA_CONFIG") or DEFAULT_CAMERA_CONFIG_PATH)

# How often the page re-runs itself while a session is in progress. Fast
# enough to feel live, slow enough that the rerun (which re-reads the status
# file and re-queries the stats) is not itself a load on a Jetson that is
# simultaneously running TensorRT inference.
LIVE_REFRESH_SECONDS = 2.0

# Badge colour per camera state.
_STATE_STYLE = {
    CameraState.PENDING.value:   ("#6b7280", "Queued"),
    CameraState.RUNNING.value:   ("#2563eb", "Processing"),
    CameraState.COMPLETED.value: ("#16a34a", "Complete"),
    CameraState.FAILED.value:    ("#dc2626", "Failed"),
    CameraState.SKIPPED.value:   ("#a16207", "Skipped"),
}

st.set_page_config(
    page_title="Multi-Camera Vehicle Trajectory Reconstruction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1560px; }
      .tj-header {
        background: linear-gradient(100deg, #0b1f36 0%, #123a63 55%, #1b5e8a 100%);
        color: #fff; padding: 20px 26px; border-radius: 12px; margin-bottom: 18px;
      }
      /* Streamlit styles h1/p directly, and those rules beat the colour
         inherited from .tj-header -- which left the title dark grey on a
         dark blue banner, effectively invisible. Stated explicitly here,
         and marked important so a theme change cannot take it back. */
      .tj-header h1 {
        margin: 0; padding: 0; font-size: 1.62rem; font-weight: 700;
        letter-spacing: .2px; color: #ffffff !important;
      }
      .tj-header p  {
        margin: 5px 0 0 0; font-size: .93rem; color: rgba(255,255,255,.86) !important;
      }
      .tj-header code {
        background: rgba(255,255,255,.16); padding: 1px 7px; border-radius: 5px;
        /* Same reason as the heading: Streamlit colours <code> green, which
           is hard to read on this banner. */
        color: #ffd54f !important;
      }
      .tj-card {
        border: 1px solid rgba(128,128,128,.24); border-radius: 10px;
        padding: 13px 15px; margin-bottom: 11px;
      }
      .tj-badge {
        display: inline-block; padding: 2px 10px; border-radius: 11px;
        font-size: .73rem; font-weight: 700; color: #fff; letter-spacing: .3px;
      }
      .tj-plate {
        font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
        font-size: 1.28rem; font-weight: 800; background: #ffd54f; color: #1a1a1a;
        padding: 6px 16px; border-radius: 6px; letter-spacing: 1.6px;
        display: inline-block;
      }
      .tj-log {
        background: #0d1117; color: #b9f6ca; font-family: ui-monospace, monospace;
        font-size: .77rem; padding: 11px 13px; border-radius: 8px;
        height: 232px; overflow-y: auto; line-height: 1.55; white-space: pre-wrap;
      }
      .tj-step { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 2px; }
      .tj-dot {
        width: 27px; height: 27px; border-radius: 50%; color: #fff; flex: none;
        font: 700 12px/27px system-ui, sans-serif; text-align: center;
      }
      .tj-line { width: 2px; height: 26px; background: #94a3b8; margin-left: 12.5px; }
      .tj-step-body { padding-top: 3px; }
      .tj-step-name { font-weight: 650; font-size: .95rem; }
      .tj-step-meta { font-size: .79rem; opacity: .72; }
      div[data-testid="stMetricValue"] { font-size: 1.42rem; }
      .tj-table-wrap { overflow-x: auto; }
      .tj-table {
        border-collapse: collapse; width: 100%; font-size: .84rem;
      }
      .tj-table th, .tj-table td {
        padding: 6px 10px; text-align: left; white-space: nowrap;
        border-bottom: 1px solid rgba(128,128,128,.22);
      }
      .tj-table th {
        font-weight: 650; opacity: .75; text-transform: uppercase;
        font-size: .72rem; letter-spacing: .4px;
      }
      .tj-table tbody tr:hover { background: rgba(128,128,128,.09); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── cached resources ──────────────────────────────────────────────────────


@st.cache_resource
def _bootstrap() -> dict:
    """Load the config and open the database once.

    `cache_resource` (not `cache_data`): these are live handles, and the
    database must be initialised exactly once per process, not once per
    rerun. The camera registry is deliberately NOT cached here -- see
    _registry().
    """
    config = load_config(str(CONFIG_PATH))
    database.init_db(config.get("database", {}).get("path", "data/alpr.db"))
    return {"config": config}


def _manager():
    """This process's CameraManager."""
    return get_camera_manager(_bootstrap()["config"], str(camera_config_path()))


def _registry():
    """The one camera registry this page reads.

    The manager's own, deliberately: adding or removing a camera mutates
    that registry, and a second copy cached beside it would leave the map,
    the heatmap and the search box describing a deployment that no longer
    exists until the next restart.
    """
    return _manager().registry


# ── cached database reads ─────────────────────────────────────────────────
#
# Every panel on this page runs at least one query, and several run the SAME
# query: the blacklist feeds both the red banner and the Home panel, and
# st.tabs renders all three tab bodies on every run, so the Operations
# statistics and the Home heatmap are computed even while nobody is looking
# at either of them. Multiplied by a live session's refresh, that was dozens
# of queries a second against the same handful of rows.
#
# cache_data with a short TTL collapses each distinct query to one call per
# interval and still refreshes fast enough to read as live. A TTL rather
# than explicit invalidation, because the PIPELINE writes these rows from
# its own threads -- there is no edit on this page to hang an invalidation
# off. Edits made HERE are the exception and clear the caches outright (see
# _clear_query_caches), so an operator's own action is never shown stale.
#
# Every one of these returns plain rows -- dicts, lists, ints -- which is
# what cache_data requires: no ORM instance outlives its session.

_QUERY_TTL = LIVE_REFRESH_SECONDS


@st.cache_data(ttl=_QUERY_TTL, show_spinner=False)
def _q_processing_sessions(limit: int = 20) -> list[dict]:
    try:
        with database.get_session() as session:
            return database.get_processing_sessions(session, limit=limit)
    except Exception:
        return []


@st.cache_data(ttl=_QUERY_TTL, show_spinner=False)
def _q_blacklisted(plates: tuple) -> list[dict]:
    """Sightings of the given blacklisted plates.

    Takes the plates as an argument rather than reading the blacklist
    itself: the list is editable on this page, and passing it in makes an
    edit a different cache key instead of a change the TTL would hide for a
    couple of seconds.
    """
    if not plates:
        return []
    try:
        with database.get_session() as session:
            return database.get_blacklisted_detections(session, set(plates))
    except Exception:
        return []


@st.cache_data(ttl=_QUERY_TTL, show_spinner=False)
def _q_session_stats(session_filter: Optional[str]) -> dict:
    with database.get_session() as session:
        return database.get_session_stats(session, processing_session=session_filter)


@st.cache_data(ttl=_QUERY_TTL, show_spinner=False)
def _q_current_locations(session_filter: Optional[str]) -> dict:
    with database.get_session() as session:
        return database.get_current_vehicle_locations(
            session, processing_session=session_filter)


@st.cache_data(ttl=_QUERY_TTL, show_spinner=False)
def _q_events_without_session() -> int:
    try:
        with database.get_session() as session:
            return database.count_events_without_session(session)
    except Exception:
        return 0


@st.cache_data(ttl=_QUERY_TTL, show_spinner=False)
def _q_plate_detections(plate: str, session_filter: Optional[str]) -> list[dict]:
    with database.get_session() as session:
        return database.get_plate_detections(
            session, plate, processing_session=session_filter)


def _clear_query_caches() -> None:
    """Drop every cached query after this page itself changes the database.

    The TTL above exists for the pipeline's writes, which this page cannot
    see coming. An operator's own edit is different: deleting a session must
    be gone from the scope picker, the statistics and the heatmap on the very
    next render, not up to _QUERY_TTL seconds later.
    """
    for query in (
        _q_processing_sessions, _q_blacklisted, _q_session_stats,
        _q_current_locations, _q_events_without_session, _q_plate_detections,
    ):
        query.clear()


# ── formatting helpers ────────────────────────────────────────────────────


# Streamlit renamed st.image's "fit to container" argument: <=1.39 spells it
# use_column_width, >=1.40 use_container_width (and deprecates the old one).
# requirements.txt pins 1.34 while this machine runs 1.39, so the dashboard has
# to work either way -- resolved once here by inspecting the real signature
# rather than by pinning the dashboard to one Streamlit release.
_IMAGE_FIT_KWARG = (
    "use_container_width"
    if "use_container_width" in inspect.signature(st.image).parameters
    else "use_column_width"
)


def _image(path: str, **kwargs) -> None:
    """st.image() sized to its container, on any supported Streamlit."""
    st.image(path, **{_IMAGE_FIT_KWARG: True}, **kwargs)


def _table(rows: list[dict], empty: str = "Nothing to show yet.") -> None:
    """Render a small table as HTML.

    Deliberately NOT st.dataframe/st.table. Both serialise through Apache
    Arrow, and this deployment pairs pyarrow 25 with numpy 1.26 -- an ABI
    mismatch (pyarrow 25 is built against numpy 2.x) that SEGFAULTS inside
    pyarrow's convert_column under some conditions. That does not raise an
    exception Streamlit can show: it kills the whole server process, so the
    operator sees "Connection error" and loses the run in progress.

    Upgrading numpy is not an option here -- requirements.txt pins 1.26.4 and
    torch, ultralytics and OpenCV are all built against it, so moving it to
    satisfy a table renderer would risk the ALPR pipeline itself. These
    tables are a handful of rows, so plain HTML costs nothing and cannot
    crash. Values are escaped; None renders as a dash.
    """
    if not rows:
        st.caption(empty)
        return

    columns = list(rows[0])
    head = "".join(f"<th>{html_escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{html_escape('--' if row.get(c) is None else str(row.get(c)))}</td>"
            for c in columns
        ) + "</tr>"
        for row in rows
    )
    st.markdown(
        f'<div class="tj-table-wrap"><table class="tj-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _fmt_time(raw: Optional[str], with_date: bool = True) -> str:
    """Render a stored UTC timestamp in the viewer's local time."""
    if not raw:
        return "--"
    try:
        parsed = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return str(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    local = parsed.astimezone()
    return local.strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S")


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def _fmt_confidence(value: Optional[float]) -> str:
    return "--" if value is None else f"{value * 100:.1f}%"


def _resolve_image(path: Optional[str]) -> Optional[Path]:
    """Resolve a stored image path against the repo root, if it exists."""
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate if candidate.is_file() else None


def _badge(text: str, color: str) -> str:
    return f'<span class="tj-badge" style="background:{color}">{text}</span>'


# ══════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════


def render_header(status: dict) -> None:
    state = status.get("state", SessionState.IDLE.value)
    label = {
        SessionState.IDLE.value: "Idle — ready to process",
        SessionState.RUNNING.value: "Processing in progress",
        SessionState.COMPLETED.value: "Last session completed",
        SessionState.FAILED.value: "Last session finished with errors",
        SessionState.CANCELLED.value: "Last session cancelled",
    }.get(state, state)

    session_id = status.get("session_id") or "—"
    st.markdown(
        f"""
        <div class="tj-header">
          <h1>Multi-Camera Vehicle Trajectory Reconstruction System</h1>
          <p>ALPR across a camera network · {label} · session <code>{session_id}</code></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
#  LEFT PANEL — camera uploads + START
# ══════════════════════════════════════════════════════════════════════════


def _describe_source(camera) -> tuple[bool, str]:
    """Return (is_usable, human description) for a camera's current source."""
    source = camera.source_uri
    if not source:
        return False, "No source set"
    if camera.source_type.value == "rtsp":
        return True, f"Stream: {mask_credentials(source)}"
    return (True, f"Video: {Path(source).name}") if Path(source).is_file() else (
        False, f"Missing file: {Path(source).name}"
    )


def _render_camera_source(camera, manager, running: bool) -> None:
    """Source controls for ONE camera: an uploaded video *or* an RTSP stream.

    A camera has exactly one source. The radio picks which kind, and the
    manager enforces the exclusivity on the data itself -- assigning a video
    clears the stream URL and vice versa -- so "which source is this camera
    on?" is always a one-field question, both here and at processing time.
    """
    camera_id = camera.camera_id
    is_rtsp = camera.source_type.value == "rtsp"

    kind = st.radio(
        f"Source for {camera.camera_name}",
        ["Upload video", "RTSP stream"],
        index=1 if is_rtsp else 0,
        key=f"kind_{camera_id}",
        horizontal=True,
        disabled=running,
        label_visibility="collapsed",
    )

    if kind.endswith("Upload video"):
        upload = st.file_uploader(
            f"Video for {camera.camera_name}",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            key=f"upload_{camera_id}",
            disabled=running,
            label_visibility="collapsed",
        )
        if upload is not None:
            # Streamlit re-delivers the same file object on every rerun, so
            # without this fingerprint the video would be rewritten to disk
            # several times a second for as long as the widget holds it.
            marker = f"{upload.name}:{upload.size}"
            if st.session_state.get(f"saved_{camera_id}") != marker:
                try:
                    manager.save_upload(camera_id, upload.name, upload.getvalue())
                    st.session_state[f"saved_{camera_id}"] = marker
                    st.rerun()
                except (ValueError, OSError, KeyError) as exc:
                    st.error(f"Upload failed: {exc}")
    else:
        # A plain input + button, deliberately NOT st.form. Inside a form the
        # submit never reached set_rtsp_url here -- the camera silently kept
        # its uploaded video and the run processed the file instead of the
        # stream, with the card still showing the URL. A button is simpler,
        # fires reliably, and is testable.
        url_key = f"rtsp_url_{camera_id}"
        st.session_state.setdefault(url_key, camera.rtsp_url or "")
        st.text_input(
            f"RTSP URL for {camera.camera_name}",
            key=url_key,
            placeholder="rtsp://192.168.1.50:554/stream1",
            label_visibility="collapsed",
            disabled=running,
        )
        check_key = f"stream_check_{camera_id}"
        buttons = st.columns(2)
        if buttons[0].button(
            "Set stream",
            key=f"set_rtsp_{camera_id}",
            use_container_width=True,
            disabled=running,
        ):
            try:
                updated = manager.set_rtsp_url(camera_id, st.session_state[url_key])
                # The uploaded-file marker is deliberately NOT cleared: the
                # file_uploader keeps holding the previous file, so clearing
                # it meant that merely switching the radio back to "Upload
                # video" re-saved that file and silently replaced the stream.
                #
                # Check reachability now. Saving is still allowed -- a camera
                # can be offline for a moment -- but the operator should find
                # out here, not three minutes into a run.
                result = probe_stream(updated.rtsp_url)
                st.session_state[check_key] = {
                    "ok": result.ok, "message": result.message, "frame": None,
                }
                st.rerun()
            except VideoSourceError as exc:
                # Validated up front, so a typo is caught here rather than
                # several minutes into a processing run.
                st.error(str(exc))
            except KeyError as exc:
                st.error(f"Unknown camera: {exc}")

        if buttons[1].button(
            "Test stream",
            key=f"test_rtsp_{camera_id}",
            use_container_width=True,
            disabled=running or not st.session_state.get(url_key),
            help="Connect to the stream and grab one frame, without running ALPR.",
        ):
            with st.spinner("Connecting to the stream…"):
                result, frame = grab_preview_frame(st.session_state[url_key])
            preview = None
            if frame is not None:
                import cv2

                ok, encoded = cv2.imencode(".jpg", frame)
                preview = encoded.tobytes() if ok else None
            st.session_state[check_key] = {
                "ok": result.ok, "message": result.message, "frame": preview,
            }

        check = st.session_state.get(check_key)
        if check:
            if check["ok"]:
                st.success(check["message"])
            else:
                st.error(check["message"])
            if check.get("frame"):
                st.image(check["frame"], caption="Preview frame from the stream",
                         **{_IMAGE_FIT_KWARG: True})


def _render_add_camera(manager, running: bool) -> None:
    """Add a camera site: place, latitude, longitude.

    Coordinates are asked for up front, not left optional, because a camera
    without them records events that no map or heatmap can place -- it would
    look configured and then quietly vanish from every view that matters.
    """
    with st.expander("Add camera", expanded=False):
        if running:
            st.caption("Stop the session before adding a camera.")
            return

        place = st.text_input(
            "Place", key="new_camera_place",
            placeholder="e.g. Rajiv Chowk",
            help="Shown on the dashboard and stored on every event this camera records.",
        )
        coordinates = st.columns(2)
        latitude = coordinates[0].text_input(
            "Latitude", key="new_camera_lat", placeholder="28.6129",
        )
        longitude = coordinates[1].text_input(
            "Longitude", key="new_camera_lon", placeholder="77.2295",
        )

        if st.button("Add camera", key="add_camera_go", use_container_width=True,
                     type="primary"):
            try:
                camera = manager.add_camera(place, latitude, longitude)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
            else:
                # Clear the form, then rerun so the new camera's own card,
                # source controls and feed panel are drawn immediately.
                for key in ("new_camera_place", "new_camera_lat", "new_camera_lon"):
                    st.session_state.pop(key, None)
                st.session_state["camera_notice"] = (
                    f"Added {camera.camera_name} ({camera.camera_id}). "
                    "Give it a video or an RTSP stream below."
                )
                st.rerun()

        st.caption(
            "The new camera behaves exactly like the others: upload or RTSP, "
            "its own feed panel, its own place on the map, heatmap and queue."
        )


def _render_rename_camera(camera, manager, running: bool) -> None:
    """Rename a camera site in place.

    Only the label changes: the camera keeps its id, coordinates, queue
    position and source, and events already recorded keep the name they were
    stamped with.
    """
    with st.expander("Rename", expanded=False):
        if running:
            st.caption("Stop the session before renaming a camera.")
            return

        name = st.text_input(
            "Place", value=camera.camera_name,
            key=f"rename_{camera.camera_id}",
            label_visibility="collapsed",
        )
        if st.button("Save name", key=f"rename_go_{camera.camera_id}",
                     use_container_width=True):
            try:
                renamed = manager.rename_camera(camera.camera_id, name)
            except (KeyError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
            else:
                st.session_state.pop(f"rename_{camera.camera_id}", None)
                st.session_state["camera_notice"] = (
                    f"{camera.camera_id} is now {renamed.camera_name}."
                )
                st.rerun()
        st.caption("Detections already recorded keep the name they were stored with.")


def _render_remove_camera(camera, manager, running: bool, container) -> None:
    """Two-step removal, so one stray click cannot delete a camera site."""
    confirm_key = f"confirm_remove_{camera.camera_id}"

    if st.session_state.get(confirm_key):
        st.warning(
            f"Remove **{camera.camera_name}** ({camera.camera_id})? Its feed "
            "panel goes too. Detections it already recorded are kept."
        )
        choice = st.columns(2)
        if choice[0].button("Remove", key=f"remove_yes_{camera.camera_id}",
                            type="primary", use_container_width=True):
            try:
                manager.remove_camera(camera.camera_id)
            except (KeyError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
                st.session_state.pop(confirm_key, None)
            else:
                st.session_state.pop(confirm_key, None)
                st.session_state.pop(f"saved_{camera.camera_id}", None)
                st.session_state["camera_notice"] = (
                    f"Removed {camera.camera_name} ({camera.camera_id})."
                )
                st.rerun()
        if choice[1].button("Cancel", key=f"remove_no_{camera.camera_id}",
                            use_container_width=True):
            st.session_state.pop(confirm_key, None)
            st.rerun()
        return

    if container.button(
        "Remove", key=f"remove_{camera.camera_id}",
        use_container_width=True, disabled=running,
        help="Remove this camera site. Its recorded detections are kept.",
    ):
        st.session_state[confirm_key] = True
        st.rerun()


def render_camera_panel(status: dict) -> None:
    st.subheader("Cameras")
    manager = _manager()
    running = manager.is_running()
    progress_by_id = {c["camera_id"]: c for c in status.get("cameras", [])}

    st.caption(
        "Each camera takes **one** source — an uploaded video **or** an RTSP "
        "stream. Setting one replaces the other."
    )

    notice = st.session_state.pop("camera_notice", None)
    if notice:
        st.success(notice)

    ready = 0
    live_cameras = 0
    # Same grid as the camera wall: CAMERAS_PER_ROW across, the rest below.
    # Stacked in one narrow column, five cameras were a metre of scrolling
    # with each card's upload box, buttons and rename control between the
    # next one -- and no way to see the whole deployment at once.
    for row in wall_rows(list(manager.registry)):
        columns = st.columns(CAMERAS_PER_ROW, gap="medium")
        for column, camera in zip(columns, row):
            progress = progress_by_id.get(camera.camera_id)
            state = progress["state"] if progress else CameraState.PENDING.value
            color, label = _STATE_STYLE.get(state, ("#6b7280", state))

            location = (
                f"{camera.latitude:.4f}, {camera.longitude:.4f}"
                if camera.has_location else "no coordinates configured"
            )
            usable, source_text = _describe_source(camera)
            if usable:
                ready += 1
                # "Runs until STOP", not "is RTSP": with loop_recorded on,
                # uploaded clips run continuously too.
                if manager.runs_continuously(camera):
                    live_cameras += 1

            with column:
                st.markdown(
                    f"""<div class="tj-card">
                          <b>{camera.order}. {camera.camera_name}</b> {_badge(label, color)}<br>
                          <span style="opacity:.72;font-size:.82rem">
                            {camera.camera_id} &middot; {location}<br>{source_text}
                          </span>
                        </div>""",
                    unsafe_allow_html=True,
                )

                _render_camera_source(camera, manager, running)

                actions = st.columns(2)
                if usable and not running:
                    if actions[0].button(
                        "Clear", key=f"clear_{camera.camera_id}",
                        use_container_width=True,
                    ):
                        manager.clear_source(camera.camera_id)
                        st.session_state.pop(f"saved_{camera.camera_id}", None)
                        st.rerun()

                _render_remove_camera(camera, manager, running, actions[1])
                _render_rename_camera(camera, manager, running)

                if progress and progress.get("error"):
                    st.caption(f"Error: {progress['error']}")

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # The Add camera form and the run controls belong to the whole section,
    # not to a column, so they sit under the grid at a readable width.
    controls = st.columns([1, 1, 2])
    with controls[0]:
        _render_add_camera(manager, running)

    st.divider()

    total = len(manager.registry.enabled)
    if manager.loops_recorded:
        schedule = (
            "Every camera runs **together** -- uploaded videos loop like live "
            "feeds -- and nothing stops until you press **STOP PROCESSING**."
        )
    else:
        schedule = (
            "Uploaded videos are processed **one at a time**, in order; "
            "RTSP streams all run **together**."
        )
    st.caption(f"**{ready} of {total}** camera(s) have a source. {schedule}")

    if live_cameras:
        # A live stream carries what is happening now, so it is not queued
        # behind anything and it is not cut short -- it runs until STOP.
        st.info(
            f"{live_cameras} camera(s) will start together and keep "
            "running until you press STOP.",
        )

    run_button, _ = st.columns([1, 3])
    if running:
        if run_button.button("STOP PROCESSING", type="secondary",
                             use_container_width=True):
            manager.stop()
            st.rerun()
    else:
        if run_button.button(
            "START PROCESSING",
            type="primary",
            use_container_width=True,
            disabled=ready == 0,
        ):
            try:
                session_id = manager.start()
                st.success(f"Session `{session_id}` started")
                time.sleep(0.4)   # let the worker publish its first status
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
        if ready == 0:
            st.caption("Give at least one camera a source to enable processing.")


# ══════════════════════════════════════════════════════════════════════════
#  CENTER — live processing status
# ══════════════════════════════════════════════════════════════════════════


def _estimate_remaining(cameras: list[dict]) -> Optional[float]:
    """Seconds of work left across the queue, or None if not yet knowable.

    Uses the throughput actually being achieved on the camera in flight,
    applied to its remaining frames plus every frame still queued behind it.
    Deliberately measured rather than assumed: frame rate varies several-fold
    with how many vehicles are in view, so a fixed per-frame estimate would
    be wrong in both directions.

    Returns None until a camera is running with a measured rate and a known
    frame count -- a live stream has no total, and a guess presented as an
    estimate is worse than no estimate.
    """
    running = next(
        (c for c in cameras if c["state"] == CameraState.RUNNING.value), None
    )
    if not running or running["fps"] <= 0 or not running.get("total_frames"):
        return None

    rate = running["fps"]
    remaining = max(0, running["total_frames"] - running["frames_processed"])
    for camera in cameras:
        if camera["state"] == CameraState.PENDING.value:
            # A queued camera's own frame count is only known once its source
            # has been probed; fall back to the running camera's as the best
            # available guide.
            remaining += camera.get("total_frames") or running["total_frames"]
    return remaining / rate


def _live_frame_bytes(camera_id: str) -> Optional[bytes]:
    """Read a camera's latest annotated frame, or None if it has none yet.

    Returned as BYTES rather than a path on purpose: Streamlit caches images
    by path, so passing the same filename every rerun would freeze the panel
    on the first frame it ever saw. The pipeline rewrites these files a few
    times a second, so the bytes are what change.

    Never raises -- the file is being rewritten underneath us, so a partial
    or vanished read is expected and simply means "no new frame this time".
    """
    frame = _live_frame_path(camera_id)
    try:
        if not frame.is_file():
            return None
        data = frame.read_bytes()
        # A JPEG caught mid-write has no end-of-image marker; showing it
        # would make Streamlit log a decode error every refresh.
        return data if data.endswith(b"\xff\xd9") else None
    except OSError:
        return None


def _live_frame_path(camera_id: str) -> Path:
    """Where the pipeline publishes this camera's latest annotated frame."""
    directory = (_bootstrap()["config"].get("api") or {}).get(
        "live_frames_dir", "data/live_frames"
    )
    path = Path(directory)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path / f"{camera_id}.jpg"


def _has_live_frame(camera_id: str) -> bool:
    """Whether a camera has published a complete frame, without reading it.

    The same question _live_frame_bytes answers, end-of-image check and all,
    but it seeks to the last two bytes instead of loading the whole JPEG.
    When the preview server is up the wall shows video and never displays
    these pixels -- the file is consulted only to decide between a caption
    and no caption, and reading a few hundred KB per camera per refresh to
    answer that was pure disk traffic.
    """
    frame = _live_frame_path(camera_id)
    try:
        if not frame.is_file():
            return False
        with frame.open("rb") as handle:
            handle.seek(-2, os.SEEK_END)
            return handle.read(2) == b"\xff\xd9"
    except OSError:
        # Includes a file shorter than two bytes, i.e. one just created.
        return False


# Height of each camera's video panel on the wall, in CSS pixels.
# The camera wall puts CAMERAS_PER_ROW feeds side by side. Four across is a
# surveillance wall rather than four big screens: the whole deployment fits
# one glance on a laptop, and a fifth camera starts the next row instead of
# pushing the first four off-screen. The panel is sized to match -- a quarter
# of the width no longer needs the height a half-width panel did.
CAMERAS_PER_ROW = 4
FEED_PANEL_HEIGHT = 168


def _preview_server():
    """The MJPEG preview server for the camera wall, started on first use."""
    api_cfg = _bootstrap()["config"].get("api") or {}
    frames_dir = Path(api_cfg.get("live_frames_dir", "data/live_frames"))
    if not frames_dir.is_absolute():
        frames_dir = REPO_ROOT / frames_dir
    return get_preview_server(
        frames_dir,
        host=str(api_cfg.get("preview_host", "0.0.0.0")),
        port=int(api_cfg.get("preview_port", 8765)),
    )


def _stream_player_html(camera_id: str, port: int) -> str:
    """A self-contained player for one camera's MJPEG preview stream.

    Depends ONLY on the camera id and the port. That is deliberate: Streamlit
    re-runs this page every couple of seconds during processing, and an
    iframe whose content changes between runs is reloaded -- restarting the
    video each time, which is the choppiness this replaces. Identical content
    lets the browser keep the same player and the same open stream, so the
    video plays continuously while the status text around it updates.

    The stream is served on its own port, so its address is built in the
    browser from the hostname the operator actually used to open the
    dashboard (localhost on the Jetson, its LAN IP from another machine).
    """
    camera_json = json.dumps(camera_id)
    return f"""
<div style="width:100%;height:{FEED_PANEL_HEIGHT}px;background:#111;border-radius:8px;overflow:hidden">
  <img id="feed" alt="Live preview"
       style="width:100%;height:100%;object-fit:contain;display:block">
</div>
<script>
(function () {{
  var img = document.getElementById("feed");
  var host = "localhost";
  try {{ host = window.parent.location.hostname || host; }}
  catch (e) {{ try {{ host = new URL(document.referrer).hostname || host; }} catch (e2) {{}} }}
  var url = "http://" + host + ":{int(port)}/stream/" + encodeURIComponent({camera_json});
  // Reconnect if the stream cannot be reached yet, instead of leaving a
  // broken-image icon in the panel.
  img.onerror = function () {{ setTimeout(connect, 2000); }};
  function connect() {{ img.src = url + "?t=" + Date.now(); }}
  connect();
}})();
</script>
"""


def _panel_status(camera, state: str, is_live: bool, progress) -> tuple[str, str]:
    """(message, css) explaining an empty panel, or ("", "") when there is none."""
    if state == CameraState.FAILED.value and progress and progress.get("error"):
        return f"Could not start: {progress['error']}", "color:#dc2626"
    if is_live and camera.source_type.value == "rtsp":
        return "Connecting to the stream…", ""
    if is_live:
        return "Starting…", ""
    if state == CameraState.SKIPPED.value:
        return "Skipped — no source set", ""
    return "Waiting for its turn in the queue", ""


def _blacklisted_sightings() -> list[dict]:
    # Sorted into a tuple so the blacklist (a set) is a stable cache key:
    # the same plates in a different iteration order must not be a miss.
    return _q_blacklisted(tuple(sorted(get_blacklist().plates())))


def render_blacklist_alert() -> None:
    """Red banner at the top of the page while any blacklisted plate has been seen."""
    sightings = _blacklisted_sightings()
    if not sightings:
        return
    latest: dict = {}
    for row in sightings:                      # newest first
        plate = row["plate_number"].upper()
        latest.setdefault(plate, {"row": row, "count": 0})["count"] += 1
    lines = []
    for plate, item in latest.items():
        row = item["row"]
        entry = get_blacklist().get(plate) or {}
        reason = f" — {entry['reason']}" if entry.get("reason") else ""
        lines.append(
            f"**{plate}**{reason}: last seen at **{row.get('camera_name') or 'unknown camera'}**, "
            f"{_fmt_time(row.get('timestamp'))} ({item['count']} sighting(s))"
        )
    st.error("**BLACKLISTED VEHICLE DETECTED**  \n" + "  \n".join(lines))


def render_blacklist_panel() -> None:
    """Manage the blacklist and see every sighting of a blacklisted vehicle."""
    st.subheader("Blacklisted vehicles")
    blacklist = get_blacklist()

    form = st.columns([2, 3, 1])
    plate_text = form[0].text_input("Plate", placeholder="DL7CD5017",
                                    label_visibility="collapsed", key="blacklist_plate")
    reason_text = form[1].text_input("Reason", placeholder="Reason (optional)",
                                     label_visibility="collapsed", key="blacklist_reason")
    if form[2].button("Add", use_container_width=True, key="blacklist_add"):
        try:
            blacklist.add(plate_text, reason_text)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    entries = blacklist.entries()
    if not entries:
        st.caption("No plates are blacklisted.")
        return

    sightings = _blacklisted_sightings()
    by_plate: dict = {}
    for row in sightings:
        by_plate.setdefault(row["plate_number"].upper(), []).append(row)

    for entry in entries:
        rows = by_plate.get(entry["plate"], [])
        cols = st.columns([1, 3, 1, 1])
        latest = rows[0] if rows else {}
        image = (_resolve_image(latest.get("vehicle_thumbnail_path"))
                 or _resolve_image(latest.get("vehicle_image_path"))
                 or _resolve_image(latest.get("image_path")))
        with cols[0]:
            if image:
                _image(str(image))
            else:
                st.caption("No image")
        with cols[1]:
            st.markdown(f'{_badge("BLACKLISTED", "#dc2626")} <span class="tj-plate">{entry["plate"]}</span>',
                        unsafe_allow_html=True)
            detail = entry["reason"] or "No reason given"
            if rows:
                st.caption(f"{detail} · {len(rows)} sighting(s) · last at "
                           f"{latest.get('camera_name') or 'unknown camera'}, {_fmt_time(latest.get('timestamp'))}")
            else:
                st.caption(f"{detail} · not detected yet")
        with cols[2]:
            if rows and st.button("Trajectory", key=f"bl_view_{entry['plate']}", use_container_width=True):
                st.session_state["active_plate"] = entry["plate"]
                st.rerun()
        with cols[3]:
            if st.button("Remove", key=f"bl_remove_{entry['plate']}", use_container_width=True):
                blacklist.remove(entry["plate"])
                st.rerun()


def render_traffic_heatmap(session_filter: Optional[str]) -> None:
    """Traffic heatmap across every camera site, for the selected scope."""
    st.subheader("Traffic heatmap")
    view = st.radio(
        "Heatmap view",
        ["Where vehicles are now", "Total traffic per camera"],
        horizontal=True,
        label_visibility="collapsed",
        key="heatmap_view",
    )
    current = view.startswith("Where")
    try:
        if current:
            # Each vehicle counted once, at the camera it was LAST seen:
            # vehicles that passed 1, 2 and 3 and are now at 4 heat up 4.
            counts = _q_current_locations(session_filter)
        else:
            stats = _q_session_stats(session_filter)
            counts = {row["camera_id"]: row["detections"] for row in stats["per_camera"]}
    except Exception as exc:
        st.warning(f"Heatmap unavailable: {exc}")
        return
    sites = [
        {"camera_id": c.camera_id, "camera_name": c.camera_name, "latitude": c.latitude,
         "longitude": c.longitude, "detections": counts.get(c.camera_id, 0)}
        for c in _registry()
    ]
    total = sum(site["detections"] for site in sites)
    scope = f"session `{session_filter}`" if session_filter else "all sessions"
    if current:
        st.caption(f"Each vehicle shown at the camera where it was last seen ({scope}): "
                   f"{total} vehicle(s). Hotter = more vehicles there now; grey = none.")
    else:
        st.caption(f"Every detection at each camera site ({scope}): {total} in total. "
                   "Hotter = more traffic passed; grey = none.")
    components.html(render_traffic_heatmap_html(sites), height=480)


def wall_rows(cameras: list, per_row: int = CAMERAS_PER_ROW) -> list[list]:
    """Group cameras into the rows of the camera wall.

    Four across, in registry order, and a fifth camera starts the next row
    rather than shrinking the first four.
    """
    return [cameras[start:start + per_row]
            for start in range(0, len(cameras), per_row)]


def render_camera_wall(status: dict) -> None:
    """All cameras' feeds in one view, four per row.

    Live (RTSP) cameras run together, so every live panel moves at once.
    Recorded files are processed in turn, so only the camera whose clip is
    playing updates and the rest hold the last frame they produced -- which
    is what makes this a summary of the whole run rather than a single moving
    picture. Each panel is labelled with the camera and its state, so a still
    image is never mistaken for a live one.
    """
    manager = _manager()
    cameras = list(manager.registry)
    if not cameras:
        return

    progress_by_id = {c["camera_id"]: c for c in status.get("cameras", [])}
    server = _preview_server()
    streaming = server.running

    st.subheader("Camera feeds")
    # No explanatory caption: the panels are labelled with their camera and
    # state, which is all this section needs to say. The one thing worth
    # interrupting for is smooth video being unavailable -- silently showing
    # stills instead would look like a broken feed -- and that is a fault,
    # not a description, so it is a warning and only appears when it is true.
    if not streaming:
        st.warning(
            f"Live video unavailable ({server.error}) — showing still frames. "
            "Free the port or change `api.preview_port` in config.yaml, then "
            "restart the dashboard."
        )

    any_frame = False
    for row in wall_rows(cameras):
        # Always CAMERAS_PER_ROW columns, even for a short last row, so a
        # fifth camera gets a panel the same size as the first four rather
        # than one stretched across the whole page.
        columns = st.columns(CAMERAS_PER_ROW, gap="small")
        for column, camera in zip(columns, row):
            with column:
                progress = progress_by_id.get(camera.camera_id)
                state = progress["state"] if progress else CameraState.PENDING.value
                color, label = _STATE_STYLE.get(state, ("#6b7280", state))
                is_live = state == CameraState.RUNNING.value
                # With no session running, "Queued" read as stuck. It is not
                # queued for anything -- it is waiting for START.
                idle = not manager.is_running() and state == CameraState.PENDING.value
                if idle:
                    label = "Ready"

                st.markdown(
                    f"**{camera.order}. {camera.camera_name}** "
                    f"{_badge('LIVE' if is_live else label, '#dc2626' if is_live else color)}",
                    unsafe_allow_html=True,
                )

                # The streaming panel needs a yes/no, the fallback needs the
                # pixels. Ask for only what this panel will actually use.
                frame = None
                if streaming:
                    has_frame = _has_live_frame(camera.camera_id)
                else:
                    frame = _live_frame_bytes(camera.camera_id)
                    has_frame = frame is not None
                if has_frame:
                    any_frame = True

                if streaming:
                    # Continuous video, independent of page re-runs.
                    components.html(
                        _stream_player_html(camera.camera_id, server.port),
                        height=FEED_PANEL_HEIGHT + 8,
                    )
                    if not has_frame:
                        message, css = _panel_status(camera, state, is_live, progress)
                        if idle:
                            message, css = "Press START PROCESSING to begin", ""
                        st.markdown(
                            f'<div style="font-size:.8rem;opacity:.8;{css}">'
                            f"{html_escape(message)}</div>",
                            unsafe_allow_html=True,
                        )
                elif frame:
                    # Fallback when the stream server could not start: a still
                    # frame refreshed on each page re-run.
                    st.image(frame, **{_IMAGE_FIT_KWARG: True})
                else:
                    message, css = _panel_status(camera, state, is_live, progress)
                    if idle:
                        message, css = "Press START PROCESSING to begin", ""
                    st.markdown(
                        '<div style="min-height:150px;border:1px dashed rgba(128,128,128,.4);'
                        'border-radius:8px;display:flex;align-items:center;'
                        'justify-content:center;text-align:center;padding:12px;'
                        f'font-size:.82rem;opacity:.75;{css}">{html_escape(message)}</div>',
                        unsafe_allow_html=True,
                    )

                source_kind = (
                    "RTSP stream" if camera.source_type.value == "rtsp" else "Recorded video"
                )
                detail = f"{camera.camera_id} · {source_kind}"
                if progress:
                    detail += f" · {progress['detections']} detection(s)"
                    if progress.get("total_frames"):
                        detail += (
                            f" · frame {progress['frames_processed']:,}"
                            f"/{progress['total_frames']:,}"
                        )
                st.caption(detail)

    if not any_frame:
        st.caption(
            "Frames appear here once processing starts — the pipeline "
            "publishes its annotated view a few times a second."
        )


def render_processing_status(status: dict) -> None:
    st.subheader("Live Processing Status")

    cameras = status.get("cameras", [])
    state = status.get("state", SessionState.IDLE.value)

    if not cameras:
        manager = _manager()
        schedule = (
            "Every camera runs together, uploaded videos looping like live "
            "feeds, until you press STOP."
            if manager.loops_recorded
            else "Uploaded videos are processed one at a time, in order; RTSP "
            "streams all run together until you stop them."
        )
        st.info(
            "No processing session yet. Give each camera a source in the "
            "**Cameras** section above, then press **START PROCESSING**. "
            + schedule
        )
        return

    current = next(
        (c for c in cameras if c["state"] == CameraState.RUNNING.value), None
    )

    # Before the first frame of the session there is genuinely nothing to
    # report: the models are still being deserialised, which takes minutes on
    # this hardware. Saying so is the difference between "working" and
    # "hung" -- the panel otherwise sits at 0% with 0.0 FPS and looks stuck.
    warming_up = (
        state == SessionState.RUNNING.value
        and not any(c["frames_processed"] for c in cameras)
    )
    if warming_up:
        latest = status.get("logs", [])
        detail = latest[-1].split("  ", 1)[-1] if latest else "Preparing…"
        st.info(
            f"Starting up — {detail}\n\n"
            "The detection and OCR models are loaded once per session "
            "(this takes a couple of minutes on this hardware) and then "
            "reused for every camera. Frame counts start moving after that.",
            icon=None,
        )

    top = st.columns([2, 1, 1])
    with top[0]:
        st.metric("Current camera", current["camera_name"] if current else "—")
    with top[1]:
        st.metric("FPS", f"{current['fps']:.1f}" if current else "—")
    with top[2]:
        st.metric("Session progress", f"{status.get('percent', 0):.0f}%")

    st.progress(
        min(1.0, max(0.0, status.get("percent", 0) / 100.0)),
        text=f"Queue: {sum(1 for c in cameras if c['state'] == CameraState.COMPLETED.value)}"
             f"/{len(cameras)} camera(s) complete",
    )

    # An honest estimate of how much longer this will take. Without it a slow
    # clip (this pipeline runs at a couple of frames per second on this
    # hardware, so a few hundred frames is minutes per camera) is
    # indistinguishable from a stall -- the operator sees a bar that barely
    # moves and assumes something is wrong.
    eta = _estimate_remaining(cameras)
    if eta is not None:
        st.caption(f"Estimated time remaining: about {_fmt_duration(eta)}.")

    # Per-camera progress bars, in queue order.
    for camera in cameras:
        color, label = _STATE_STYLE.get(camera["state"], ("#6b7280", camera["state"]))
        head = st.columns([3, 1, 1, 1])
        head[0].markdown(
            f"**{camera['order']}. {camera['camera_name']}** {_badge(label, color)}",
            unsafe_allow_html=True,
        )
        head[1].caption(f"{camera['detections']} detection(s)")
        head[2].caption(f"{camera['unique_plates']} plate(s)")
        head[3].caption(_fmt_duration(camera.get("elapsed_seconds")))
        # An indeterminate source (a live stream) reports total_frames=0 and
        # therefore percent=0 while running -- show the bar anyway so the
        # camera does not look stalled.
        st.progress(min(1.0, max(0.0, camera.get("percent", 0) / 100.0)))

    st.divider()

    detail = st.columns([1, 1, 2])
    if current and current.get("total_frames"):
        st.caption(
            f"{current['camera_name']}: frame {current['frames_processed']:,} "
            f"of {current['total_frames']:,} at {current['fps']:.1f} fps"
        )

    with detail[0]:
        st.markdown("**Current plate**")
        last_plate = current.get("last_plate") if current else None
        if last_plate:
            st.markdown(f'<span class="tj-plate">{last_plate}</span>', unsafe_allow_html=True)
        else:
            st.caption("Awaiting first detection…")
    with detail[1]:
        st.markdown("**Current vehicle**")
        image = _resolve_image(
            (current or {}).get("last_vehicle_image")
            or (current or {}).get("last_plate_image")
        )
        if image:
            _image(str(image))
            details = [
                value for value in (
                    ((current or {}).get("last_vehicle_class") or "").title(),
                    (current or {}).get("last_vehicle_color"),
                ) if value
            ]
            if details:
                st.caption(" · ".join(details))
        else:
            st.caption("No image yet")
    with detail[2]:
        st.markdown("**Processing log**")
        logs = status.get("logs", [])[-60:]
        body = "\n".join(reversed(logs)) if logs else "Waiting for the session to start…"
        st.markdown(f'<div class="tj-log">{body}</div>', unsafe_allow_html=True)

    if state == SessionState.FAILED.value and status.get("error"):
        st.error(f"Session error: {status['error']}")


# ══════════════════════════════════════════════════════════════════════════
#  RIGHT — statistics
# ══════════════════════════════════════════════════════════════════════════


def render_statistics(status: dict, session_filter: Optional[str]) -> None:
    """Detection counts for the scope the operator has chosen.

    Scoped by the sidebar's session picker -- the same filter the trajectory
    view uses -- so the two panels can never disagree about which run is
    being looked at.

    This deliberately does NOT scope to the status file's session id. That
    id is whatever ran last on this machine, which may belong to a different
    database (a demo DB, a restored backup) or to a run whose events have
    since been cleared; scoping to it then reported 0 detections while the
    trajectory view, using the sidebar filter, happily showed the data.
    """
    st.subheader("Statistics")

    session_id = session_filter
    try:
        stats = _q_session_stats(session_id)
    except Exception as exc:
        st.warning(f"Statistics unavailable: {exc}")
        return

    cameras = status.get("cameras", [])
    running = [c for c in cameras if c["state"] == CameraState.RUNNING.value]

    # Averaged across cameras that have actually reported a figure, so a
    # queue that has only run two of four cameras is not dragged toward zero
    # by the two that have not started.
    reported_ocr = [c["avg_ocr_ms"] for c in cameras if c.get("avg_ocr_ms")]
    reported_det = [c["avg_detection_ms"] for c in cameras if c.get("avg_detection_ms")]

    # One row across the page. Stacked in pairs they were a tall narrow
    # column, which is what the right-hand column used to force.
    row = st.columns(6)
    row[0].metric("Vehicles detected", stats["total_detections"])
    row[1].metric("Unique plates", stats["unique_plates"])
    row[2].metric("Processing time", _fmt_duration(status.get("elapsed_seconds")))
    row[3].metric("Current camera", running[0]["camera_name"] if running else "—")
    row[4].metric(
        "Avg OCR time",
        f"{sum(reported_ocr) / len(reported_ocr):.1f} ms" if reported_ocr else "—",
    )
    row[5].metric(
        "Avg detection time",
        f"{sum(reported_det) / len(reported_det):.1f} ms" if reported_det else "—",
    )

    if not session_filter:
        outside = _q_events_without_session()
        if outside:
            st.caption(
                f"{outside} older single-gate event(s) are not counted here — "
                "they are not part of any multi-camera session. They are "
                "still shown in the original dashboard."
            )

    st.markdown("**Detections per camera**")
    if session_filter:
        st.caption(f"Scoped to session `{session_filter}`.")
    if stats["per_camera"]:
        _table([
            {
                "Camera": row_["camera_name"] or row_["camera_id"] or "—",
                "Detections": row_["detections"],
                "Plates": row_["unique_plates"],
                "Avg conf": _fmt_confidence(row_["avg_confidence"]),
            }
            for row_ in stats["per_camera"]
        ])
    else:
        st.caption("No detections recorded for this session yet.")


# ══════════════════════════════════════════════════════════════════════════
#  SEARCH RESULT — summary, timeline, map, history
# ══════════════════════════════════════════════════════════════════════════


def _load_trajectory(plate: str, session_filter: Optional[str]):
    registry = _registry()
    engine = TrajectoryEngine.from_registry(registry)
    # Only the query is cached. The engine builds from rows already in
    # memory, and its result holds live registry objects that have no
    # business being pickled into a cache.
    detections = _q_plate_detections(plate, session_filter)
    return engine.build(plate, detections, processing_session=session_filter), detections


def render_trajectory_view(plate: str, session_filter: Optional[str]) -> None:
    try:
        trajectory, detections = _load_trajectory(plate, session_filter)
    except Exception as exc:
        st.error(f"Could not reconstruct a trajectory for {plate}: {exc}")
        return

    if trajectory.is_empty:
        st.warning(
            f"No detections found for **{plate}**"
            + (f" in session `{session_filter}`." if session_filter else ".")
        )
        return

    # ── summary ───────────────────────────────────────────────────────────
    try:
        with database.get_session() as db_session:
            profile = get_profile(db_session, trajectory.plate_number) or {}
    except Exception:
        profile = {}

    head = st.columns([1, 3])
    with head[0]:
        best = max(
            trajectory.points,
            key=lambda p: p.confidence if p.confidence is not None else -1.0,
        )
        # Profile thumbnails first (the clearest read across all detections),
        # then the full-size crops this card always used.
        vehicle_image = (_resolve_image(profile.get("vehicle_thumbnail_path"))
                         or _resolve_image(best.vehicle_image_path))
        plate_image = (_resolve_image(profile.get("plate_thumbnail_path"))
                       or _resolve_image(best.plate_image_path))
        if vehicle_image:
            _image(str(vehicle_image), caption=f"Best read — {best.camera_name}")
        if plate_image:
            _image(str(plate_image), caption="Plate")
        if not vehicle_image and not plate_image:
            st.info("No vehicle image stored for this plate.")
    with head[1]:
        badge = _badge("BLACKLISTED", "#dc2626") + " " if get_blacklist().contains(trajectory.plate_number) else ""
        st.markdown(f'{badge}<span class="tj-plate">{trajectory.plate_number}</span>', unsafe_allow_html=True)
        attributes = []
        if profile.get("vehicle_class"):
            attributes.append(f"**Vehicle type:** {profile['vehicle_class'].title()}")
        if profile.get("vehicle_color"):
            attributes.append(f"**Vehicle color:** {profile['vehicle_color']}")
        if attributes:
            st.markdown(" &nbsp;·&nbsp; ".join(attributes), unsafe_allow_html=True)
        st.write("")
        row = st.columns(3)
        row[0].metric("Cameras visited", trajectory.cameras_visited)
        row[1].metric("Total detections", trajectory.total_detections)
        row[2].metric("Avg confidence", _fmt_confidence(trajectory.average_confidence))
        row = st.columns(3)
        row[0].metric("First seen", _fmt_time(trajectory.first_seen, with_date=False))
        row[1].metric("Last seen", _fmt_time(trajectory.last_seen, with_date=False))
        row[2].metric("Travel time", _fmt_duration(trajectory.duration_seconds))
        if trajectory.total_distance_km is not None:
            st.caption(
                f"Path length **{trajectory.total_distance_km:.2f} km** across "
                f"{len(trajectory.points)} stop(s) · ordered by *{trajectory.ordering}*"
            )

    st.divider()

    # ── timeline + map ────────────────────────────────────────────────────
    left, right = st.columns([1, 2.1])

    with left:
        st.markdown("### Trajectory timeline")
        count = len(trajectory.points)
        for index, point in enumerate(trajectory.points):
            # Same green->red ramp the map polyline uses, so a point is
            # recognisable between the two views.
            hue = 145 - (145 * index / max(1, count - 1)) if count > 1 else 145
            st.markdown(
                f"""<div class="tj-step">
                      <div class="tj-dot" style="background:hsl({hue:.0f},78%,44%)">{point.sequence}</div>
                      <div class="tj-step-body">
                        <div class="tj-step-name">{point.camera_name}</div>
                        <div class="tj-step-meta">
                          {_fmt_time(point.timestamp)} ·
                          {_fmt_confidence(point.confidence)} ·
                          {point.detection_count} read(s)
                        </div>
                      </div>
                    </div>""",
                unsafe_allow_html=True,
            )
            if index < count - 1:
                leg = trajectory.legs[index] if index < len(trajectory.legs) else None
                caption = ""
                if leg is not None:
                    bits = []
                    if leg.distance_km is not None:
                        bits.append(f"{leg.distance_km:.2f} km")
                    if leg.duration_seconds is not None:
                        bits.append(_fmt_duration(leg.duration_seconds))
                    if leg.speed_kmh is not None:
                        bits.append(f"{leg.speed_kmh:.0f} km/h")
                    caption = " · ".join(bits)
                st.markdown(
                    f'<div class="tj-line"></div>'
                    f'<div style="margin-left:38px;margin-top:-24px;margin-bottom:6px;'
                    f'font-size:.75rem;opacity:.62">{caption}</div>',
                    unsafe_allow_html=True,
                )

    with right:
        st.markdown("### Route map")
        satellite = st.toggle(
            "Satellite view",
            value=False,
            key=f"sat_{plate}",
            help="Esri World Imagery with place labels. Toggle layers on the map itself too.",
        )
        unmapped = len(trajectory.points) - len(trajectory.mappable_points)
        if unmapped:
            st.caption(
                f"{unmapped} point(s) are not plotted — those cameras have "
                "no coordinates in camera_config.yaml."
            )
        components.html(
            render_trajectory_map(
                trajectory,
                repo_root=REPO_ROOT,
                height=520,
                satellite_default=satellite,
            ),
            height=540,
            scrolling=False,
        )

    st.divider()

    # ── history table ─────────────────────────────────────────────────────
    st.markdown("### Detection history")
    st.caption(
        f"Every stored read for this plate ({len(detections)} row(s)) — the "
        "timeline above collapses repeated reads at one camera into a single visit."
    )
    _table([
        {
            "Camera": row.get("camera_name") or row.get("camera_id") or "—",
            "Camera ID": row.get("camera_id") or "—",
            "Time": _fmt_time(row.get("timestamp")),
            "Confidence": _fmt_confidence(row.get("confidence")),
            "Latitude": f"{row['latitude']:.5f}" if row.get("latitude") is not None else None,
            "Longitude": f"{row['longitude']:.5f}" if row.get("longitude") is not None else None,
            "Direction": row.get("direction") or "—",
            "Session": row.get("processing_session") or "—",
            "Source": Path(row["video_source"]).name if row.get("video_source") else "—",
        }
        for row in detections
    ])

    st.download_button(
        "Download trajectory GeoJSON",
        data=json.dumps(trajectory_to_geojson(trajectory), indent=2, default=str),
        file_name=f"trajectory_{trajectory.plate_number}.geojson",
        mime="application/geo+json",
    )


# ══════════════════════════════════════════════════════════════════════════
#  SEARCH PANEL
# ══════════════════════════════════════════════════════════════════════════


def render_search_panel() -> Optional[str]:
    """Plate search box. Returns the plate to display, if any."""
    st.subheader("Search vehicle by plate number")

    form = st.columns([3, 1, 2])
    with form[0]:
        typed = st.text_input(
            "Plate number",
            placeholder="e.g. DL8CA1234",
            label_visibility="collapsed",
            key="search_input",
        )
    with form[1]:
        submitted = st.button("Search", type="primary", use_container_width=True)
    with form[2]:
        # Suggestions come from the database, so the operator never has to
        # guess: these are the plates that actually crossed more than one
        # camera and therefore have a path worth drawing.
        try:
            with database.get_session() as session:
                candidates = database.get_multi_camera_plates(session, min_cameras=2, limit=25)
        except Exception:
            candidates = []
        if candidates:
            choice = st.selectbox(
                "Vehicles seen at 2+ cameras",
                ["—"] + [
                    f"{c['plate_number']}  ({c['camera_count']} cameras)"
                    for c in candidates
                ],
                label_visibility="collapsed",
            )
            if choice != "—":
                st.session_state["active_plate"] = choice.split()[0]
        else:
            st.caption("No multi-camera vehicles yet.")

    if submitted and typed.strip():
        st.session_state["active_plate"] = typed.strip().upper()

    render_vehicle_search()
    return st.session_state.get("active_plate")


def render_vehicle_search() -> None:
    """Describe a vehicle in words; see every matching vehicle's picture.

    "white cars", "red truck after 9am today", "blue car at India Gate last 2
    hours". The sentence is parsed on this device (search/nl_query.py) into
    colour, type, time window, camera and plate, and run against the vehicle
    profiles. What was understood is shown above the results, so a surprising
    match is explainable.
    """
    st.subheader("Search vehicles by description")
    form = st.columns([4, 1])
    with form[0]:
        text = st.text_input(
            "Describe the vehicle",
            placeholder="e.g. white cars after 9am today · commercial trucks with yellow plate · cars seen at 3+ cameras · leaving India Gate",
            label_visibility="collapsed",
            key="vehicle_search_input",
        )
    with form[1]:
        if st.button("Find vehicles", use_container_width=True, key="vehicle_search_go"):
            st.session_state["vehicle_search_text"] = text.strip()

    asked = st.session_state.get("vehicle_search_text")
    if not asked:
        return

    registry = _registry()
    cameras = [(c.camera_id, c.camera_name) for c in registry]
    query = parse_query(asked, cameras=cameras)
    if query.is_empty:
        st.warning(
            "Couldn't find a vehicle feature in that (colour, type, plate, plate colour, "
            "registration, BH series, direction, cameras visited, confidence, time or camera). "
            "Try something like “white cars today” or “red truck at India Gate”."
        )
        return

    try:
        with database.get_session() as db_session:
            results = search_profiles(db_session, query, limit=60)
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return

    st.caption(f"Understood: {query.describe(dict(cameras))} — {len(results)} vehicle(s)")
    if not results:
        st.info("No stored vehicles match.")
        return

    for row_start in range(0, len(results), 4):
        columns = st.columns(4)
        for column, vehicle in zip(columns, results[row_start:row_start + 4]):
            with column:
                image = (_resolve_image(vehicle.get("vehicle_thumbnail_path"))
                         or _resolve_image(vehicle.get("vehicle_image_path"))
                         or _resolve_image(vehicle.get("plate_thumbnail_path")))
                if image:
                    _image(str(image))
                else:
                    st.caption("No image stored")
                visit = vehicle.get("matched_visit") or {}
                label = " ".join(v for v in (vehicle.get("vehicle_color"),
                                             (vehicle.get("vehicle_class") or "").title()) if v)
                flag = (_badge("BLACKLISTED", "#dc2626") + " "
                        if get_blacklist().contains(vehicle["plate_number"]) else "")
                st.markdown(f"{flag}**{html_escape(vehicle['plate_number'])}**  \n"
                            f"{html_escape(label or 'Unknown')}", unsafe_allow_html=True)
                st.caption(f"{visit.get('camera_name') or '—'} · {_fmt_time(visit.get('last_seen'))}")
                if st.button("View trajectory", key=f"nl_pick_{vehicle['plate_number']}",
                             use_container_width=True):
                    st.session_state["active_plate"] = vehicle["plate_number"]
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
#  LIVE PANELS — the parts that move on their own while a session runs
# ══════════════════════════════════════════════════════════════════════════
#
# Each of the four panels that changes by itself during a run is paired with
# a fragment version of itself. st.experimental_fragment(run_every=...)
# re-runs JUST that panel on a timer, so a live session repaints the camera
# wall and the progress bars without re-executing the sidebar, the session
# queries, all three tab bodies, the heatmap iframe and the trajectory view
# along with them.
#
# This replaces a `time.sleep(LIVE_REFRESH_SECONDS); st.rerun()` that used
# to sit at the end of main(). That re-ran the whole script every couple of
# seconds to move a progress bar, and -- worse -- the sleep held the script
# thread for the entire interval, so a click was not acted on until the nap
# finished. That delay, not the rendering, is what made the page feel stuck.
#
# None of these takes a status argument. A fragment re-runs WITHOUT main(),
# so it has to read the current status itself rather than close over the one
# main() read when the page was last built in full.


# Set by main() while a run is live, so that the header knows a stand-down
# is owed when the run finishes and an idle page never asks for one.
_LIVE_ARMED = "_live_panels_armed"


def _panel_header() -> None:
    manager = _manager()
    running = manager.is_running()
    render_header(manager.get_status())
    # A fragment timer keeps firing for as long as the page is open, so
    # something has to notice the run has ended and put main() back in
    # charge. The header does it because it is the one panel always on
    # screen, and only once -- disarming first, so a page left open idle
    # does not re-run itself every couple of seconds forever.
    if not running and st.session_state.get(_LIVE_ARMED):
        st.session_state[_LIVE_ARMED] = False
        st.rerun()


def _panel_camera_wall() -> None:
    render_camera_wall(_manager().get_status())


def _panel_processing_status() -> None:
    render_processing_status(_manager().get_status())


def _panel_statistics(session_filter: Optional[str]) -> None:
    render_statistics(_manager().get_status(), session_filter)


# Each panel exists twice: once with a refresh timer, once without. Both
# are built from the SAME function, and that is the entire point.
#
# Streamlit identifies a fragment by md5(module + qualname + container
# path), and it only records that id when the fragment is actually called
# during a FULL script run. A timer already ticking in the browser is then
# resolved against those ids. So if a full run draws the page WITHOUT the
# fragment -- which is exactly what happens the moment a session finishes
# and the page goes back to its static form -- an in-flight timer arrives
# for an id that no longer exists, and Streamlit kills the run with
# "RuntimeError: Could not find fragment with id ...". That is what used to
# greet anyone whose run ended, including a run that failed on startup.
#
# Because both variants are decorated from the same function, they hash to
# the same id. The page can move between live and static as often as it
# likes and a late timer always lands on something real.
def _with_timer(panel):
    return st.experimental_fragment(run_every=LIVE_REFRESH_SECONDS)(panel)


def _without_timer(panel):
    return st.experimental_fragment()(panel)


_panel_header_auto = _with_timer(_panel_header)
_panel_header_static = _without_timer(_panel_header)
_panel_camera_wall_auto = _with_timer(_panel_camera_wall)
_panel_camera_wall_static = _without_timer(_panel_camera_wall)
_panel_processing_status_auto = _with_timer(_panel_processing_status)
_panel_processing_status_static = _without_timer(_panel_processing_status)
_panel_statistics_auto = _with_timer(_panel_statistics)
_panel_statistics_static = _without_timer(_panel_statistics)


def _autostart_once() -> str:
    """Start processing the first time this dashboard process renders a page.

    The once-per-process guard lives in cameras.manager (see autostart_once
    there for why it cannot be a Streamlit cache). An environment variable on
    the dashboard container turns it on, not a config key: the camera config
    is also read by the API, CLI tools and the tests, and a test that rendered
    this page with the real config used to start a real GPU session.
    """
    from src.cameras.manager import autostart_once

    enabled = os.environ.get("ALPR_AUTOSTART", "").strip().lower() in {"1", "true", "yes"}
    return autostart_once(_manager(), enabled)


def main() -> None:
    try:
        _bootstrap()
    except CameraConfigError as exc:
        st.error(f"Camera configuration problem: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Startup failed: {exc}")
        st.stop()

    manager = _manager()
    _autostart_once()
    status = manager.get_status()

    # The auto-refresh checkbox lives in the sidebar, which is built further
    # down, but the header is the first thing drawn and has to know whether
    # to draw itself as a self-refreshing fragment. Its value is therefore
    # read from session state; on the very first run, before the widget
    # exists, that falls back to the checkbox's own default.
    live = st.session_state.get("auto_refresh", True) and manager.is_running()

    if live:
        st.session_state[_LIVE_ARMED] = True
    (_panel_header_auto if live else _panel_header_static)()
    render_blacklist_alert()

    # ── sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")
        st.checkbox(
            "Auto-refresh while processing", value=True, key="auto_refresh",
            help="Refreshes the live panels every couple of seconds during "
                 "a run. Keyed so the header, which is drawn before this "
                 "sidebar, can read it too.",
        )
        st.divider()

        st.subheader("Session scope")
        sessions = _q_processing_sessions()
        options = ["All sessions"] + [
            f"{s['processing_session']}  ({s['detections']} events)" for s in sessions
        ]
        picked = st.selectbox(
            "Reconstruct trajectories from", options,
            help=(
                "Each run of the camera queue is one session. Scope to a single "
                "run to keep repeated demos of the same video from blending together."
            ),
        )
        session_filter = None if picked == "All sessions" else picked.split()[0]

        # ── delete a session ──────────────────────────────────────────────
        # Two-step on purpose: this removes stored detections and their
        # images and cannot be undone, so a single mis-click on a narrow
        # sidebar control must not be enough to destroy a run.
        if sessions:
            deletable = st.selectbox(
                "Delete a session",
                ["—"] + [
                    f"{row['processing_session']}  ({row['detections']} events)"
                    for row in sessions
                ],
                help=(
                    "Permanently removes every event recorded in that run, "
                    "and the plate and vehicle crops they reference."
                ),
            )
            if deletable != "—":
                target = deletable.split()[0]
                confirm_key = f"confirm_delete_{target}"
                if not st.session_state.get(confirm_key):
                    if st.button(
                        f"Delete session {target}",
                        use_container_width=True,
                        disabled=manager.is_running(),
                    ):
                        st.session_state[confirm_key] = True
                        st.rerun()
                else:
                    st.warning(
                        f"Delete **{target}** and all its events? "
                        "This cannot be undone."
                    )
                    confirm_columns = st.columns(2)
                    if confirm_columns[0].button(
                        "Yes, delete", type="primary", use_container_width=True
                    ):
                        try:
                            counts = delete_processing_session(
                                _bootstrap()["config"], target
                            )
                            # The database is done; now the live status,
                            # header and camera wall must stop describing
                            # the run that was just deleted.
                            manager.forget_session(target)
                            # Those rows are gone; nothing may keep serving
                            # them from cache for the rest of the TTL.
                            _clear_query_caches()
                            st.session_state.pop(confirm_key, None)
                            # A trajectory on screen may have just lost its
                            # underlying events.
                            if st.session_state.get("active_plate"):
                                st.session_state.pop("active_plate", None)
                            st.success(
                                f"Deleted {counts['events']} event(s) and "
                                f"{counts['images']} image(s)."
                            )
                            time.sleep(0.8)
                            st.rerun()
                        except (ValueError, RuntimeError) as exc:
                            st.error(f"Delete failed: {exc}")
                    if confirm_columns[1].button("Cancel", use_container_width=True):
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
            if manager.is_running():
                st.caption("Sessions cannot be deleted while a run is in progress.")

        st.divider()
        st.subheader("Camera network")
        for camera in manager.registry:
            mark = "on" if camera.enabled else "off"
            coords = (
                f"{camera.latitude:.4f}, {camera.longitude:.4f}"
                if camera.has_location else "not surveyed"
            )
            st.caption(f"**{camera.camera_id}** &middot; {camera.camera_name} ({mark})\n\n{coords}")

        st.divider()
        st.caption(
            "Configure cameras in `config/camera_config.yaml`. "
            "To go live, set a camera's `source_type` to `rtsp` and fill in "
            "`rtsp_url` — nothing else changes."
        )

        # The registry is parsed once per process and the manager is a
        # module-level singleton (both deliberate -- Streamlit re-runs this
        # script on every interaction, and rebuilding either would drop a
        # running worker). The cost is that an edit to camera_config.yaml is
        # invisible until something drops those caches, which is what this
        # does. Refuses while a session is running: swapping the registry
        # under a queue mid-run would leave the worker processing cameras
        # that no longer match the status it is publishing.
        if st.button(
            "Reload camera config",
            use_container_width=True,
            disabled=manager.is_running(),
            help="Re-read config/camera_config.yaml without restarting the dashboard.",
        ):
            try:
                reset_camera_manager()
                _bootstrap.clear()
                st.session_state.pop("bulk_upload_saved", None)
                st.success("Camera configuration reloaded.")
                st.rerun()
            except CameraConfigError as exc:
                st.error(f"Camera configuration problem: {exc}")
        if manager.is_running():
            st.caption("Reload is disabled while a session is running.")

    # ── pages ─────────────────────────────────────────────────────────────
    #
    # Home is the watch page: what is happening, and how to look something
    # up. Operations is the control page: sources, the run, and what it
    # produced. Trajectory is one vehicle's route. Splitting them keeps the
    # page somebody leaves open all day from scrolling past four columns of
    # controls they only touch when starting a run.
    tab_home, tab_ops, tab_traj = st.tabs(["Home", "Operations", "Trajectory"])

    with tab_home:
        render_blacklist_panel()

        st.divider()
        (_panel_camera_wall_auto if live else _panel_camera_wall_static)()

        st.divider()
        plate = render_search_panel()
        if plate:
            st.info(f"Showing **{plate}** on the **Trajectory** tab.")

        # Last: the heatmap is the one section that is read rather than used,
        # so putting it below the search boxes keeps both of those within
        # reach of the feeds instead of behind a full-width map.
        st.divider()
        render_traffic_heatmap(session_filter)

    with tab_ops:
        # Stacked, each at full width: the cameras are a grid of their own
        # (four across), so squeezing them into a third of the page was what
        # made five cameras an unreadable column.
        render_camera_panel(status)

        st.divider()
        (_panel_processing_status_auto if live else _panel_processing_status_static)()

        st.divider()
        (_panel_statistics_auto if live else _panel_statistics_static)(session_filter)

    with tab_traj:
        plate = st.session_state.get("active_plate")
        if not plate:
            st.info(
                "Search for a plate on the **Home** tab to reconstruct its "
                "route across the camera network."
            )
        else:
            header = st.columns([4, 1])
            header[0].markdown(f"## Trajectory — `{plate}`")
            if header[1].button("Clear", use_container_width=True):
                st.session_state.pop("active_plate", None)
                st.rerun()
            render_trajectory_view(plate, session_filter)

    # No refresh loop here any more. While a session runs, the four live
    # panels above re-run themselves on their own timers (see _live_fragment)
    # and this function is left alone, so the page stays responsive to
    # clicks instead of sleeping between repaints.


if __name__ == "__main__":
    main()
