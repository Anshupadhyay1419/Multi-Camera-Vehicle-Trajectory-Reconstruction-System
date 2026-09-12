"""
FastAPI routes for the multi-camera trajectory system.

All mounted under /trajectory-api so they cannot collide with the existing
single-gate routes (/entry, /logs, /search, ...) or with the static
dashboard mounted at "/". Every existing endpoint keeps its path and its
behaviour; this module only adds.

  GET  /trajectory-api/cameras                camera registry
  POST /trajectory-api/cameras/upload         upload every camera's video at once
  POST /trajectory-api/cameras/{id}/upload    upload one camera's video
  POST /trajectory-api/cameras/{id}/rtsp      switch a camera to RTSP
  POST /trajectory-api/processing/start       start the sequential run
  POST /trajectory-api/processing/stop        request a stop
  GET  /trajectory-api/processing/status      live progress
  GET  /trajectory-api/trajectory/{plate}     reconstructed path
  GET  /trajectory-api/map/{plate}            path as GeoJSON
  GET  /trajectory-api/map/{plate}/html       path as a Leaflet page
  GET  /trajectory-api/statistics             session/camera aggregates
  GET  /trajectory-api/plates                 plates worth plotting
  GET  /trajectory-api/sessions               recent processing sessions
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from src.api.schemas import (
    CameraResponse,
    RTSPRequest,
    StartProcessingRequest,
    StartProcessingResponse,
    TrajectoryResponse,
)
from src.cameras.registry import CameraConfigError, load_camera_registry
from src.cameras.status_store import StatusStore
from src.database import db as database
from src.trajectory import TrajectoryEngine, trajectory_to_geojson
from src.utils.logger import get_logger

_logger = get_logger("api.trajectory_routes")
_REPO_ROOT = Path(__file__).resolve().parents[2]

router = APIRouter(prefix="/trajectory-api", tags=["multi-camera"])

# Largest video the API will accept, to keep one bad request from filling
# the Jetson's disk. Read fully into memory before writing, so this is also
# the memory ceiling per upload.
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


# ── lazily-built shared state ─────────────────────────────────────────────
#
# The registry is parsed once per process and cached. The CameraManager is
# built only when something actually asks to process or to see status --
# constructing it is cheap, but the pipeline it eventually imports is not,
# and a read-only request for the camera list should not pay for that.

_registry = None
_manager = None


def _get_registry():
    global _registry
    if _registry is None:
        try:
            _registry = load_camera_registry(
                str(_REPO_ROOT / "config" / "camera_config.yaml")
            )
        except CameraConfigError as exc:
            _logger.error("Camera configuration unavailable: %s", exc)
            raise HTTPException(
                status_code=503, detail=f"Camera configuration unavailable: {exc}"
            )
    return _registry


def _get_manager():
    global _manager
    if _manager is None:
        from src.cameras.manager import CameraManager
        from src.utils.config import load_config

        try:
            config = load_config(str(_REPO_ROOT / "config" / "config.yaml"))
        except Exception as exc:
            _logger.error("Could not load config.yaml for the camera manager: %s", exc)
            raise HTTPException(status_code=503, detail=f"Configuration unavailable: {exc}")
        _manager = CameraManager(config=config, registry=_get_registry())
    return _manager


def _get_engine() -> TrajectoryEngine:
    """A TrajectoryEngine configured from the registry's `trajectory:` block.

    Built per request rather than cached: it is a three-field object with no
    connections to hold, and rebuilding it means a config edit takes effect
    without restarting the API.
    """
    return TrajectoryEngine.from_registry(_get_registry())


def _reconstruct(plate: str, processing_session: Optional[str]):
    """Load a plate's detections and reconstruct its path.

    Shared by every endpoint that returns a trajectory in some encoding, so
    the three of them cannot drift apart in how they order or collapse
    points.
    """
    try:
        with database.get_session() as session:
            detections = database.get_plate_detections(
                session, plate, processing_session=processing_session
            )
    except RuntimeError as exc:
        _logger.error("Database unavailable reconstructing %s: %s", plate, exc)
        raise HTTPException(status_code=503, detail="Database unavailable")

    return _get_engine().build(plate, detections, processing_session=processing_session)


# ── camera configuration ──────────────────────────────────────────────────


@router.get("/cameras", response_model=list[CameraResponse])
def list_cameras():
    """Return every configured camera, in processing order."""
    return [CameraResponse(**camera.to_dict()) for camera in _get_registry()]


@router.post("/cameras/{camera_id}/upload")
async def upload_camera_video(camera_id: str, file: UploadFile = File(...)):
    """Store an uploaded video and point *camera_id* at it for this session.

    Each camera gets its own file even when the same recording is uploaded
    four times -- the cameras stay independent, exactly as they would be
    with four real streams.
    """
    manager = _get_manager()
    if manager.is_running():
        raise HTTPException(
            status_code=409,
            detail="A processing session is running; stop it before changing sources",
        )

    try:
        data = await file.read()
    except Exception as exc:
        _logger.error("Failed to read upload for %s: %s", camera_id, exc)
        raise HTTPException(status_code=400, detail=f"Could not read upload: {exc}")

    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {_MAX_UPLOAD_BYTES // (1024**3)} GB limit",
        )

    try:
        path = manager.save_upload(camera_id, file.filename or "", data)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No camera {camera_id!r}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        _logger.error("Failed to store upload for %s: %s", camera_id, exc)
        raise HTTPException(status_code=500, detail=f"Could not store upload: {exc}")

    return {
        "camera_id": camera_id,
        "video_path": str(path),
        "size_bytes": len(data),
        "message": f"Video stored for {camera_id}",
    }


@router.post("/cameras/upload")
async def upload_camera_videos(files: list[UploadFile] = File(...)):
    """Upload every camera's footage in one request.

    The bulk counterpart to the per-camera route above, and what the
    dashboard's single uploader calls. One file is broadcast to all enabled
    cameras (the four-virtual-cameras demo); several are assigned in queue
    order. Either way each camera receives its own copy on disk, so no two
    cameras share a video_path.
    """
    manager = _get_manager()
    if manager.is_running():
        raise HTTPException(
            status_code=409,
            detail="A processing session is running; stop it before changing sources",
        )

    uploads: list[tuple[str, bytes]] = []
    total_bytes = 0
    for upload in files:
        try:
            data = await upload.read()
        except Exception as exc:
            _logger.error("Failed to read upload %r: %s", upload.filename, exc)
            raise HTTPException(
                status_code=400, detail=f"Could not read {upload.filename!r}: {exc}"
            )
        total_bytes += len(data)
        # Checked against the running total, not per file: the limit exists to
        # protect the Jetson's disk and memory, and ten files just under the
        # cap would defeat a per-file check.
        if total_bytes > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Uploads exceed the {_MAX_UPLOAD_BYTES // (1024**3)} GB limit",
            )
        uploads.append((upload.filename or "", data))

    try:
        saved = manager.assign_uploads(uploads)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except OSError as exc:
        _logger.error("Failed to store bulk upload: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not store uploads: {exc}")

    return {
        "assigned": {camera_id: str(path) for camera_id, path in saved.items()},
        "files_received": len(uploads),
        "cameras_assigned": len(saved),
        "total_bytes": total_bytes,
        "message": (
            f"{len(uploads)} file(s) assigned to {len(saved)} camera(s)"
            + (" (one video broadcast to every camera)" if len(uploads) == 1 else "")
        ),
    }


@router.post("/cameras/{camera_id}/rtsp", response_model=CameraResponse)
def set_camera_rtsp(camera_id: str, request: RTSPRequest):
    """Switch a camera to a live RTSP stream for this session.

    The migration path in one call: nothing else about the camera, the
    pipeline or the dashboard changes. Persist it in camera_config.yaml to
    survive a restart.
    """
    manager = _get_manager()
    if manager.is_running():
        raise HTTPException(
            status_code=409,
            detail="A processing session is running; stop it before changing sources",
        )
    try:
        camera = manager.set_rtsp_url(camera_id, request.rtsp_url)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No camera {camera_id!r}")
    return CameraResponse(**camera.to_dict())


# ── processing control ────────────────────────────────────────────────────


@router.post("/processing/start", response_model=StartProcessingResponse)
def start_processing(request: StartProcessingRequest):
    """Start the sequential run. Returns as soon as the queue is scheduled."""
    manager = _get_manager()
    try:
        queue = manager.build_queue(request.camera_ids)
        session_id = manager.start(
            camera_ids=request.camera_ids, session_id=request.session_id
        )
    except RuntimeError as exc:
        # Already running, or nothing to run -- both are the caller's state
        # problem, not a server fault.
        raise HTTPException(status_code=409, detail=str(exc))

    return StartProcessingResponse(
        session_id=session_id,
        queued_cameras=[camera.camera_id for camera in queue],
        message=(
            f"Processing {len(queue)} camera(s) sequentially: "
            + " -> ".join(camera.camera_name for camera in queue)
        ),
    )


@router.post("/processing/stop")
def stop_processing():
    """Request that the running session stop after the current frame."""
    stopped = _get_manager().stop()
    return {
        "stopped": stopped,
        "message": (
            "Stop requested; the current camera will finish its flush"
            if stopped else "No processing session is running"
        ),
    }


@router.get("/processing/status")
def processing_status():
    """Live progress of the current or most recent session.

    Falls back to the status file when this process did not start the run --
    the dashboard and the API are usually separate processes.
    """
    try:
        return _get_manager().get_status()
    except HTTPException:
        # The manager needs config.yaml; the status file does not. A missing
        # config should not hide a run that is genuinely in progress.
        stored = StatusStore(
            str(_REPO_ROOT / _get_registry().processing.get(
                "status_file", "data/processing_status.json"
            ))
        ).read()
        if stored is None:
            raise
        return stored


# ── trajectory + map ──────────────────────────────────────────────────────


@router.get("/trajectory/{plate}", response_model=TrajectoryResponse)
def get_trajectory(
    plate: str,
    session: Optional[str] = Query(None, description="Restrict to one processing session"),
):
    """Reconstruct where a vehicle went, as ordered points and legs."""
    trajectory = _reconstruct(plate, session)
    if trajectory.is_empty:
        raise HTTPException(
            status_code=404, detail=f"No detections for plate {plate.upper()}"
        )
    return TrajectoryResponse(**trajectory.to_dict())


@router.get("/map/{plate}")
def get_map_data(
    plate: str,
    session: Optional[str] = Query(None, description="Restrict to one processing session"),
):
    """The trajectory as GeoJSON: one LineString plus one Point per visit.

    Returned even when nothing is mappable (an empty FeatureCollection), so
    a map client never has to special-case the response shape.
    """
    return JSONResponse(trajectory_to_geojson(_reconstruct(plate, session)))


@router.get("/map/{plate}/html", response_class=HTMLResponse)
def get_map_html(
    plate: str,
    session: Optional[str] = Query(None, description="Restrict to one processing session"),
    satellite: bool = Query(False, description="Open on satellite imagery"),
):
    """The trajectory as a standalone Leaflet page, ready to embed."""
    from src.mapping import render_trajectory_map

    trajectory = _reconstruct(plate, session)
    return HTMLResponse(
        render_trajectory_map(
            trajectory, repo_root=_REPO_ROOT, satellite_default=satellite
        )
    )


# ── statistics ────────────────────────────────────────────────────────────


@router.get("/statistics")
def get_statistics(
    session: Optional[str] = Query(None, description="Restrict to one processing session"),
):
    """Detection and unique-plate counts, overall and per camera."""
    try:
        with database.get_session() as db_session:
            return database.get_session_stats(db_session, processing_session=session)
    except RuntimeError as exc:
        _logger.error("Database unavailable on /statistics: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")


@router.get("/plates")
def list_trackable_plates(
    min_cameras: int = Query(2, ge=1, description="Minimum distinct cameras"),
    session: Optional[str] = Query(None, description="Restrict to one processing session"),
    limit: int = Query(100, ge=1, le=1000),
):
    """Plates seen at enough cameras to have a path worth drawing."""
    try:
        with database.get_session() as db_session:
            return database.get_multi_camera_plates(
                db_session,
                min_cameras=min_cameras,
                processing_session=session,
                limit=limit,
            )
    except RuntimeError as exc:
        _logger.error("Database unavailable on /plates: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")


@router.get("/sessions")
def list_sessions(limit: int = Query(20, ge=1, le=200)):
    """Recent processing sessions, newest first."""
    try:
        with database.get_session() as db_session:
            return database.get_processing_sessions(db_session, limit=limit)
    except RuntimeError as exc:
        _logger.error("Database unavailable on /sessions: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
