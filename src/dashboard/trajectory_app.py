"""
Multi-Camera Vehicle Trajectory Reconstruction System -- Streamlit dashboard.

Run with:
    streamlit run src/dashboard/trajectory_app.py

This is the multi-camera console. The original single-gate dashboard is
untouched and still runs (`streamlit run src/dashboard/app.py`); the two
read the same database and neither depends on the other.

Layout
    HEADER          title + live session state
    OPERATIONS tab  LEFT   per-camera upload + status, START PROCESSING
                    CENTER live processing status (camera, progress, plate,
                           vehicle, FPS, logs)
                    RIGHT  statistics
                    BOTTOM search panel
    TRAJECTORY tab  search result: vehicle, summary, timeline, map,
                    history table

Processing runs on a background thread owned by the CameraManager singleton
(see cameras/manager.py). Streamlit re-runs this script top to bottom on
every interaction, so nothing here may own long-lived state: the manager
lives at module scope in its own module, and progress is read back from the
shared status file.
"""

from __future__ import annotations

import inspect
import json
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
from src.cameras.models import CameraState, SessionState
from src.cameras.registry import CameraConfigError, load_camera_registry
from src.cameras.sources import VideoSourceError
from src.database import db as database
from src.mapping import render_trajectory_map
from src.trajectory import TrajectoryEngine, trajectory_to_geojson
from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
CAMERA_CONFIG_PATH = REPO_ROOT / "config" / "camera_config.yaml"

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
      .tj-header h1 { margin: 0; font-size: 1.62rem; font-weight: 700; letter-spacing: .2px; }
      .tj-header p  { margin: 5px 0 0 0; opacity: .82; font-size: .93rem; }
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
    """Load config, open the database, and read the camera registry once.

    `cache_resource` (not `cache_data`): these are live handles, and the
    database must be initialised exactly once per process, not once per
    rerun.
    """
    config = load_config(str(CONFIG_PATH))
    database.init_db(config.get("database", {}).get("path", "data/alpr.db"))
    registry = load_camera_registry(str(CAMERA_CONFIG_PATH))
    return {"config": config, "registry": registry}


def _manager():
    """This process's CameraManager, sharing the bootstrapped registry."""
    return get_camera_manager(_bootstrap()["config"], str(CAMERA_CONFIG_PATH))


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
          <p>ALPR across a camera network · {label} · session <code
             style="color:#ffd54f">{session_id}</code></p>
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
        return True, f"Stream: {source}"
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
        if st.button(
            "Set stream",
            key=f"set_rtsp_{camera_id}",
            use_container_width=True,
            disabled=running,
        ):
            try:
                manager.set_rtsp_url(camera_id, st.session_state[url_key])
                # The uploaded-file marker is deliberately NOT cleared: the
                # file_uploader keeps holding the previous file, so clearing
                # it meant that merely switching the radio back to "Upload
                # video" re-saved that file and silently replaced the stream.
                st.rerun()
            except VideoSourceError as exc:
                # Validated up front, so a typo is caught here rather than
                # several minutes into a processing run.
                st.error(str(exc))
            except KeyError as exc:
                st.error(f"Unknown camera: {exc}")


def render_camera_panel(status: dict) -> None:
    st.subheader("Cameras")
    manager = _manager()
    running = manager.is_running()
    progress_by_id = {c["camera_id"]: c for c in status.get("cameras", [])}

    st.caption(
        "Each camera takes **one** source — an uploaded video **or** an RTSP "
        "stream. Setting one replaces the other."
    )

    ready = 0
    live_cameras = 0
    for camera in manager.registry:
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
            if camera.source_type.value == "rtsp":
                live_cameras += 1

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

        if usable and not running:
            if st.button(
                "Clear source", key=f"clear_{camera.camera_id}",
                use_container_width=True,
            ):
                manager.clear_source(camera.camera_id)
                st.session_state.pop(f"saved_{camera.camera_id}", None)
                st.rerun()

        if progress and progress.get("error"):
            st.caption(f"Error: {progress['error']}")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    st.divider()

    total = len(manager.registry.enabled)
    st.caption(
        f"**{ready} of {total}** camera(s) have a source. "
        "Cameras are processed **one at a time**, in order."
    )

    if live_cameras:
        # A live stream has no end, so the queue needs a dwell time or it
        # would never reach the next camera. Say so plainly rather than
        # letting the run look stuck.
        dwell = manager.live_duration_seconds
        if dwell:
            st.info(
                f"{live_cameras} live stream(s): each is sampled for "
                f"**{dwell:.0f}s**, then the queue moves on.",
            )
        elif total > 1:
            st.warning(
                "A live camera with no time limit will run until stopped, so "
                "the cameras after it never run. Set "
                "`processing.live_duration_seconds` in camera_config.yaml.",
            )

    if running:
        if st.button("STOP PROCESSING", type="secondary", use_container_width=True):
            manager.stop()
            st.rerun()
    else:
        if st.button(
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
    directory = (_bootstrap()["config"].get("api") or {}).get(
        "live_frames_dir", "data/live_frames"
    )
    path = Path(directory)
    if not path.is_absolute():
        path = REPO_ROOT / path
    frame = path / f"{camera_id}.jpg"
    try:
        if not frame.is_file():
            return None
        data = frame.read_bytes()
        # A JPEG caught mid-write has no end-of-image marker; showing it
        # would make Streamlit log a decode error every refresh.
        return data if data.endswith(b"\xff\xd9") else None
    except OSError:
        return None


def render_camera_wall(status: dict) -> None:
    """All cameras' feeds in one view, two per row.

    Cameras are processed one at a time, so at most one panel is live at any
    moment; the others hold the last frame that camera produced, which is
    what makes this a useful summary of the whole run rather than a single
    moving picture. Each panel is labelled with the camera and its state, so
    a still image is never mistaken for a live one.
    """
    manager = _manager()
    cameras = list(manager.registry)
    if not cameras:
        return

    progress_by_id = {c["camera_id"]: c for c in status.get("cameras", [])}

    st.subheader("Camera feeds")
    st.caption(
        "Annotated frames from the ALPR pipeline — boxes are tracked "
        "vehicles, green once a plate has been stored. Cameras run one at a "
        "time, so only the active camera updates; the rest hold their last "
        "frame."
    )

    any_frame = False
    for row_start in range(0, len(cameras), 2):
        columns = st.columns(2, gap="medium")
        for column, camera in zip(columns, cameras[row_start:row_start + 2]):
            with column:
                progress = progress_by_id.get(camera.camera_id)
                state = progress["state"] if progress else CameraState.PENDING.value
                color, label = _STATE_STYLE.get(state, ("#6b7280", state))
                is_live = state == CameraState.RUNNING.value

                st.markdown(
                    f"**{camera.order}. {camera.camera_name}** "
                    f"{_badge('LIVE' if is_live else label, '#dc2626' if is_live else color)}",
                    unsafe_allow_html=True,
                )

                frame = _live_frame_bytes(camera.camera_id)
                if frame:
                    any_frame = True
                    st.image(frame, **{_IMAGE_FIT_KWARG: True})
                else:
                    st.markdown(
                        '<div style="height:150px;border:1px dashed rgba(128,128,128,.4);'
                        'border-radius:8px;display:flex;align-items:center;'
                        'justify-content:center;opacity:.55;font-size:.82rem">'
                        'No frames yet</div>',
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
        st.info(
            "No processing session yet. Upload a video for each camera on the "
            "left, then press **START PROCESSING**. Cameras run sequentially: "
            "each one finishes completely before the next begins."
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
        with database.get_session() as session:
            stats = database.get_session_stats(session, processing_session=session_id)
    except Exception as exc:
        st.warning(f"Statistics unavailable: {exc}")
        return

    cameras = status.get("cameras", [])
    running = [c for c in cameras if c["state"] == CameraState.RUNNING.value]

    row = st.columns(2)
    row[0].metric("Vehicles detected", stats["total_detections"])
    row[1].metric("Unique plates", stats["unique_plates"])

    row = st.columns(2)
    row[0].metric("Processing time", _fmt_duration(status.get("elapsed_seconds")))
    row[1].metric("Current camera", running[0]["camera_name"] if running else "—")

    # Averaged across cameras that have actually reported a figure, so a
    # queue that has only run two of four cameras is not dragged toward zero
    # by the two that have not started.
    reported_ocr = [c["avg_ocr_ms"] for c in cameras if c.get("avg_ocr_ms")]
    reported_det = [c["avg_detection_ms"] for c in cameras if c.get("avg_detection_ms")]
    row = st.columns(2)
    row[0].metric(
        "Avg OCR time",
        f"{sum(reported_ocr) / len(reported_ocr):.1f} ms" if reported_ocr else "—",
    )
    row[1].metric(
        "Avg detection time",
        f"{sum(reported_det) / len(reported_det):.1f} ms" if reported_det else "—",
    )

    st.divider()
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
    registry = _bootstrap()["registry"]
    engine = TrajectoryEngine.from_registry(registry)
    with database.get_session() as session:
        detections = database.get_plate_detections(
            session, plate, processing_session=session_filter
        )
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
    head = st.columns([1, 3])
    with head[0]:
        best = max(
            trajectory.points,
            key=lambda p: p.confidence if p.confidence is not None else -1.0,
        )
        image = _resolve_image(best.vehicle_image_path) or _resolve_image(best.plate_image_path)
        if image:
            _image(str(image), caption=f"Best read — {best.camera_name}")
        else:
            st.info("No vehicle image stored for this plate.")
    with head[1]:
        st.markdown(f'<span class="tj-plate">{trajectory.plate_number}</span>', unsafe_allow_html=True)
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

    return st.session_state.get("active_plate")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════


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
    status = manager.get_status()
    render_header(status)

    # ── sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")
        auto_refresh = st.checkbox(
            "Auto-refresh while processing", value=True,
            help="Re-reads live status every couple of seconds during a run.",
        )
        st.divider()

        st.subheader("Session scope")
        try:
            with database.get_session() as session:
                sessions = database.get_processing_sessions(session, limit=20)
        except Exception:
            sessions = []
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

    # ── tabs ──────────────────────────────────────────────────────────────
    tab_ops, tab_traj = st.tabs(["Operations", "Trajectory"])

    with tab_ops:
        left, center, right = st.columns([1.05, 2.0, 1.15], gap="medium")
        with left:
            render_camera_panel(status)
        with center:
            render_processing_status(status)
        with right:
            render_statistics(status, session_filter)

        st.divider()
        render_camera_wall(status)

        st.divider()
        plate = render_search_panel()
        if plate:
            st.info(f"Showing **{plate}** on the **Trajectory** tab.")

    with tab_traj:
        plate = st.session_state.get("active_plate")
        if not plate:
            st.info(
                "Search for a plate on the **Operations** tab to reconstruct its "
                "route across the camera network."
            )
        else:
            header = st.columns([4, 1])
            header[0].markdown(f"## Trajectory — `{plate}`")
            if header[1].button("Clear", use_container_width=True):
                st.session_state.pop("active_plate", None)
                st.rerun()
            render_trajectory_view(plate, session_filter)

    # Re-run the page while a session is in progress. Placed last so the
    # whole UI is painted before the sleep, and gated on the manager still
    # running so an idle dashboard costs nothing.
    if auto_refresh and manager.is_running():
        time.sleep(LIVE_REFRESH_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()
