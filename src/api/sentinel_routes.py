"""Camera Registry REST API (Sentinel Module 1, M1.2).

Mounted under /api/v1 on the EXISTING ALPR FastAPI application. This module
is an adapter and nothing more: it translates HTTP into calls on
`sentinel_system.registry.CameraService` and translates that service's
domain errors back into status codes. Every rule it appears to enforce --
unique codes, coordinate ranges, protocol/URL agreement, PTZ consistency --
actually lives in M1.1 and is enforced identically whether the caller
arrives over HTTP or imports the service directly.

Reused from the existing project rather than rebuilt:

  * the FastAPI app itself (src/api/server.py) -- this is an APIRouter that
    is included there, alongside the trajectory router
  * logging (src.utils.logger.get_logger), same naming scheme
  * credential masking (src.cameras.stream_probe.mask_credentials), so a
    stream URL never reaches a log with its password intact
  * the error convention: HTTPException with a `detail` body, which is what
    every existing route already produces
  * the database: the registry lives in the SAME database the ALPR API
    resolves, wired in server.py's lifespan, not a second one alongside it

RESPONSE SHAPE -- a deliberate decision. The brief offered a
{"success": true, "data": ...} envelope "if the existing project has no
convention". It has one: every route in this application returns the model
itself and signals failure through HTTP status plus `detail`. Wrapping only
these five endpoints would mean one API with two response shapes, and would
break the FastAPI-generated clients and the Swagger models that the
existing conventions give for free. So responses are the model, errors are
`{"detail": {"code": ..., "message": ...}}` -- the existing envelope, with
the stable machine-readable code the brief asks for carried inside it.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Path,
    Query,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.cameras.stream_probe import mask_credentials
from src.utils.logger import get_logger

from sentinel_system.core.config import get_settings
from sentinel_system.core.database import get_session
from sentinel_system.bulk_import import (
    CameraImportService,
    ImportJobNotFoundError,
    ImportReport,
    build_template_csv,
)
from sentinel_system.portal import (
    AuthExpiredError,
    AuthenticationError,
    CameraUnavailableError,
    CatalogueSync,
    NoSupportedStreamError,
    PermissionDeniedError,
    PortalError,
    PortalNotConfiguredError,
    PortalTimeoutError,
    REQUIRED_INFORMATION,
    ResilientPortalSession,
    StreamOfflineError,
    UnconfiguredPortalClient,
    load_credentials,
    select_stream,
)
from sentinel_system.verification import CameraVerificationService, VerificationResult
from sentinel_system.registry import (
    CameraCreate,
    CameraFilter,
    CameraNotFoundError,
    CameraRead,
    CameraService,
    CameraStatus,
    CameraType,
    CameraUpdate,
    DuplicateCameraCodeError,
    HealthStatus,
    Protocol,
)

_logger = get_logger("api.sentinel_routes")

router = APIRouter(prefix="/api/v1", tags=["sentinel-camera-registry"])


# ── plumbing ──────────────────────────────────────────────────────────────


def get_registry_session() -> Any:
    """Yield a registry session. Overridable in tests via dependency_overrides."""
    yield from get_session()


def get_camera_service(
    session: Annotated[Session, Depends(get_registry_session)],
) -> CameraService:
    return CameraService(session)


ServiceDep = Annotated[CameraService, Depends(get_camera_service)]


def get_verification_service(
    session: Annotated[Session, Depends(get_registry_session)],
) -> CameraVerificationService:
    return CameraVerificationService(session)


VerifyDep = Annotated[CameraVerificationService, Depends(get_verification_service)]


def get_import_service(
    session: Annotated[Session, Depends(get_registry_session)],
) -> CameraImportService:
    return CameraImportService(session)


ImportDep = Annotated[CameraImportService, Depends(get_import_service)]

#: Largest spreadsheet the importer will read. A file of cameras is
#: kilobytes; anything near this is a mistake or an attempt to exhaust the
#: server, and both are better answered before parsing than after.
MAX_IMPORT_BYTES = 25 * 1024 * 1024


async def _read_upload(file: UploadFile) -> bytes:
    """Read an upload, refusing anything oversized."""
    content = await file.read()
    if len(content) > MAX_IMPORT_BYTES:
        raise _error(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "import_file_too_large",
            f"The file is {len(content) / 1e6:.1f} MB; the limit is "
            f"{MAX_IMPORT_BYTES / 1e6:.0f} MB.",
        )
    if not content:
        raise _error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "import_file_empty",
            "The uploaded file is empty.",
        )
    return content


def _error(status_code: int, code: str, message: str) -> HTTPException:
    """An HTTPException whose detail carries a stable, machine-readable code.

    Existing routes pass a bare string as `detail`. A dict is still a valid
    `detail`, so this stays inside the same convention while giving callers
    something to branch on that is not a human-readable sentence liable to
    be reworded.
    """
    return HTTPException(
        status_code=status_code, detail={"code": code, "message": message}
    )


def _safe(camera: CameraRead | CameraCreate) -> str:
    """A camera's stream URL, with any password masked, for logging."""
    return mask_credentials(getattr(camera, "stream_url", "") or "")


class CameraPageResponse(BaseModel):
    """One page of cameras, in page/page_size terms.

    The service layer pages in limit/offset because that is what SQL does.
    HTTP callers count pages, so the translation happens here rather than
    leaking either vocabulary into the other.
    """

    items: list[CameraRead]
    total: int = Field(description="Cameras matching the filter, ignoring paging")
    page: int = Field(description="1-based page number")
    page_size: int
    total_pages: int
    has_more: bool

    model_config = {
        "json_schema_extra": {
            "example": {
                "items": [],
                "total": 128,
                "page": 1,
                "page_size": 20,
                "total_pages": 7,
                "has_more": True,
            }
        }
    }


_NOT_FOUND = {
    "description": "No camera with that id",
    "content": {
        "application/json": {
            "example": {
                "detail": {
                    "code": "camera_not_found",
                    "message": "No camera with id 3f2b...",
                }
            }
        }
    },
}
_CONFLICT = {
    "description": "camera_code already registered",
    "content": {
        "application/json": {
            "example": {
                "detail": {
                    "code": "duplicate_camera_code",
                    "message": "Camera code 'AHM-SAT-0142' is already registered",
                }
            }
        }
    },
}
_UNPROCESSABLE = {
    "description": "Field values are individually valid but contradict each other",
    "content": {
        "application/json": {
            "example": {
                "detail": {
                    "code": "invalid_camera_configuration",
                    "message": (
                        "stream_url scheme 'https' does not match protocol "
                        "'rtsp'; expected one of rtsp, rtsps"
                    ),
                }
            }
        }
    },
}


# ── endpoints ─────────────────────────────────────────────────────────────


@router.post(
    "/cameras",
    response_model=CameraRead,
    status_code=status.HTTP_201_CREATED,
    summary="Register a camera",
    responses={409: _CONFLICT, 422: _UNPROCESSABLE},
    description=(
        "Add a camera to the registry.\n\n"
        "`camera_code` is normalised to upper case and must be unique; "
        "registering a code that already exists returns **409**. Coordinates, "
        "the protocol/`stream_url` agreement and the PTZ rules are validated "
        "before anything is written, returning **422** with the offending "
        "field named."
    ),
)
def create_camera(payload: CameraCreate, service: ServiceDep) -> CameraRead:
    try:
        camera = service.create(payload)
    except DuplicateCameraCodeError as exc:
        _logger.warning("Camera rejected: %s", exc.message)
        raise _error(status.HTTP_409_CONFLICT, exc.code, exc.message) from exc
    except ValueError as exc:
        _logger.warning("Camera rejected: invalid configuration: %s", exc)
        raise _error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "invalid_camera_configuration",
            str(exc),
        ) from exc
    _logger.info(
        "Camera created: %s (%s) district=%s stream=%s",
        camera.camera_code, camera.id, camera.district, _safe(camera),
    )
    return camera


@router.get(
    "/cameras",
    response_model=CameraPageResponse,
    summary="List cameras",
    description=(
        "Paginated, filtered, sorted and searchable listing.\n\n"
        "`search` matches `camera_code` or `camera_name`, case-insensitively, "
        "treating `%` and `_` literally. Filters combine with AND.\n\n"
        "Decommissioned cameras are hidden unless `status=decommissioned` is "
        "asked for explicitly, or `include_decommissioned=true` is set -- a "
        "soft delete that still appeared in every listing would not be a "
        "delete at all."
    ),
)
def list_cameras(
    service: ServiceDep,
    department: str | None = Query(None, description="Exact match"),
    district: str | None = Query(None, description="Exact match"),
    zone: str | None = Query(None, description="Exact match"),
    vendor: str | None = Query(None, description="Exact match"),
    status_: CameraStatus | None = Query(
        None, alias="status", description="Administrative lifecycle state"
    ),
    health: HealthStatus | None = Query(None, description="Last observed condition"),
    camera_type: CameraType | None = Query(None),
    protocol: Protocol | None = Query(None),
    supports_analytics: bool | None = Query(None),
    search: str | None = Query(None, max_length=120, description="Code or name"),
    include_decommissioned: bool = Query(False),
    page: int = Query(1, ge=1, description="1-based"),
    page_size: int = Query(20, ge=1, le=200),
    sort_by: str = Query("camera_code", description="Restricted to an allow-list"),
    descending: bool = Query(False),
) -> CameraPageResponse:
    try:
        filters = CameraFilter(
            department=department,
            district=district,
            zone=zone,
            vendor=vendor,
            status=status_,
            health=health,
            camera_type=camera_type,
            protocol=protocol,
            supports_analytics=supports_analytics,
            search=search,
            include_decommissioned=include_decommissioned,
            limit=page_size,
            offset=(page - 1) * page_size,
            order_by=sort_by,
            descending=descending,
        )
    except ValueError as exc:
        # Almost always sort_by outside the allow-list. Rejected rather than
        # interpolated, because order_by reaches an ORDER BY clause.
        _logger.warning("Camera listing rejected: %s", exc)
        raise _error(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_query", str(exc)
        ) from exc

    result = service.list(filters)
    total_pages = (result.total + page_size - 1) // page_size if result.total else 0
    return CameraPageResponse(
        items=result.items,
        total=result.total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_more=result.has_more,
    )


@router.get(
    "/cameras/{camera_id}",
    response_model=CameraRead,
    summary="Get one camera",
    responses={404: _NOT_FOUND},
    description="Fetch a single camera by its UUID, decommissioned or not.",
)
def get_camera(
    service: ServiceDep,
    camera_id: uuid.UUID = Path(description="The camera's UUID"),
) -> CameraRead:
    try:
        return service.get(camera_id)
    except CameraNotFoundError as exc:
        raise _error(status.HTTP_404_NOT_FOUND, exc.code, exc.message) from exc


@router.patch(
    "/cameras/{camera_id}",
    response_model=CameraRead,
    summary="Update a camera",
    responses={404: _NOT_FOUND, 422: _UNPROCESSABLE},
    description=(
        "Partial update. Only the fields present in the body are touched: "
        "omitting a field leaves it alone, and sending `null` clears it.\n\n"
        "`camera_code` cannot be changed -- it is printed on the hardware and "
        "quoted in incident reports, so re-pointing it at different hardware "
        "would silently invalidate historical references. Retire the record "
        "and register a new one instead.\n\n"
        "Rules that need the stored row to evaluate -- setting "
        "`camera_type=ptz` on a camera whose `supports_ptz` is false, or a "
        "`stream_url` that contradicts the stored `protocol` -- return **422**."
    ),
)
def update_camera(
    payload: CameraUpdate,
    service: ServiceDep,
    camera_id: uuid.UUID = Path(description="The camera's UUID"),
) -> CameraRead:
    try:
        camera = service.update(camera_id, payload)
    except CameraNotFoundError as exc:
        raise _error(status.HTTP_404_NOT_FOUND, exc.code, exc.message) from exc
    except ValueError as exc:
        _logger.warning("Camera %s update rejected: %s", camera_id, exc)
        raise _error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "invalid_camera_configuration",
            str(exc),
        ) from exc
    _logger.info(
        "Camera updated: %s (%s) fields=%s",
        camera.camera_code, camera.id, sorted(payload.changed_fields()),
    )
    return camera


@router.delete(
    "/cameras/{camera_id}",
    response_model=None,
    summary="Retire a camera (soft delete)",
    responses={404: _NOT_FOUND, 403: {"description": "Hard delete is not enabled"}},
    description=(
        "Retires the camera by setting `status` to `decommissioned`. The "
        "record survives, because incident reports and (from M2) stored "
        "detections reference it; destroying the row would strand those "
        "references. It leaves the default listing, so the effect looks like "
        "a delete, and `?status=decommissioned` still finds it.\n\n"
        "Idempotent: retiring an already-retired camera returns **200**.\n\n"
        "`?hard=true` destroys the row instead, and is refused with **403** "
        "unless `SENTINEL_ALLOW_HARD_DELETE=true` is set on the server."
    ),
)
def delete_camera(
    service: ServiceDep,
    response: Response,
    camera_id: uuid.UUID = Path(description="The camera's UUID"),
    hard: bool = Query(
        False, description="Permanently destroy the row (requires server opt-in)"
    ),
) -> CameraRead | None:
    if hard:
        if not get_settings().allow_hard_delete:
            _logger.warning(
                "Hard delete refused for %s: SENTINEL_ALLOW_HARD_DELETE is off",
                camera_id,
            )
            raise _error(
                status.HTTP_403_FORBIDDEN,
                "hard_delete_disabled",
                "Permanent deletion is disabled; omit ?hard=true to retire "
                "the camera instead.",
            )
        try:
            service.delete(camera_id)
        except CameraNotFoundError as exc:
            raise _error(status.HTTP_404_NOT_FOUND, exc.code, exc.message) from exc
        _logger.warning("Camera permanently deleted: %s", camera_id)
        response.status_code = status.HTTP_204_NO_CONTENT
        return None

    try:
        camera = service.decommission(camera_id)
    except CameraNotFoundError as exc:
        raise _error(status.HTTP_404_NOT_FOUND, exc.code, exc.message) from exc
    _logger.info("Camera decommissioned: %s (%s)", camera.camera_code, camera.id)
    return camera


@router.post(
    "/cameras/{camera_id}/verify",
    response_model=VerificationResult,
    summary="Verify a camera",
    responses={404: _NOT_FOUND},
    description=(
        "Opens the camera's stream once, reads a few frames, measures what "
        "arrived, stores one thumbnail, and closes the connection.\n\n"
        "Runs only when asked. This is **not** health monitoring: nothing is "
        "scheduled, nothing runs in the background, and nothing keeps the "
        "stream open afterwards.\n\n"
        "A camera that cannot be reached is a **200** carrying "
        "`verified: false`, the reason in `connection_status`, and the "
        "operator-facing detail in `errors` -- not an HTTP error. Only an "
        "unregistered `camera_id` is a **404**. That split keeps 'this "
        "camera is broken' distinguishable from 'this endpoint is broken'.\n\n"
        "Measured values are stored separately from the camera's registered "
        "specification and never overwrite it: a camera commissioned as "
        "1080p25 that verifies at 704x576 is the finding, and folding one "
        "into the other would erase it.\n\n"
        "Bounded by `SENTINEL_VERIFICATION_TIMEOUT` (default 12s)."
    ),
)
def verify_camera(
    service: VerifyDep,
    camera_id: uuid.UUID = Path(description="The camera's UUID"),
) -> VerificationResult:
    try:
        return service.verify(camera_id)
    except CameraNotFoundError as exc:
        raise _error(status.HTTP_404_NOT_FOUND, exc.code, exc.message) from exc


@router.get(
    "/cameras/{camera_id}/verifications",
    response_model=list[VerificationResult],
    summary="Past verifications of a camera",
    responses={404: _NOT_FOUND},
    description=(
        "Verification history, newest first. Answers 'was this camera "
        "working last week?', which is the first question asked when one "
        "goes dark."
    ),
)
def list_verifications(
    service: VerifyDep,
    registry: ServiceDep,
    camera_id: uuid.UUID = Path(description="The camera's UUID"),
    limit: int = Query(20, ge=1, le=200),
) -> list[VerificationResult]:
    # Ask the registry first, so an unknown camera is a 404 rather than an
    # empty list that looks like "registered but never verified".
    try:
        registry.get(camera_id)
    except CameraNotFoundError as exc:
        raise _error(status.HTTP_404_NOT_FOUND, exc.code, exc.message) from exc
    return service.history(camera_id, limit=limit)


# ── bulk import ───────────────────────────────────────────────────────────
#
# Route order matters here. Starlette matches in registration order, so
# /import/template is declared BEFORE /import/{job_id} -- otherwise
# "template" is captured as a job id and the download 422s on a UUID parse.


@router.get(
    "/cameras/import/template",
    summary="Download the import template",
    response_class=Response,
    responses={
        200: {
            "description": "A CSV with the expected headings and one example row",
            "content": {"text/csv": {}},
        }
    },
    description=(
        "A CSV carrying every column the importer understands, required ones "
        "first, plus one filled-in example row.\n\n"
        "The columns are generated from the camera schema itself, so a field "
        "added to the registry cannot quietly go missing from the template. "
        "The example row is there because an empty template makes every "
        "format a guess -- what a bearing looks like, that booleans are "
        "`true`/`false`, that resolution is `WIDTHxHEIGHT`."
    ),
)
def download_import_template() -> Response:
    return Response(
        content=build_template_csv(),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="camera_import_template.csv"'
        },
    )


@router.post(
    "/cameras/import",
    response_model=ImportReport,
    summary="Bulk import cameras",
    responses={413: {"description": "File too large"}, 422: {"description": "Empty file"}},
    description=(
        "Upload a **CSV**, **Excel (.xlsx)** or **JSON** file of cameras. The "
        "format is detected from the content first and the extension second, "
        "so a spreadsheet saved as `.csv` is still read correctly.\n\n"
        "Every row goes through exactly the same schema and service call as "
        "`POST /api/v1/cameras`. There is no second, more lenient path into "
        "the registry.\n\n"
        "**Partial success is the normal outcome.** One bad row never aborts "
        "the import: good rows are committed, bad rows are reported with "
        "their file line number and the offending column, and the response "
        "carries `imported` / `failed` / `skipped` counts.\n\n"
        "`?dry_run=true` validates without creating anything -- including the "
        "already-registered check the real import would make -- so the report "
        "tells you what a commit would do."
    ),
)
async def import_cameras(
    service: ImportDep,
    file: UploadFile = File(description="A .csv, .xlsx or .json file of cameras"),
    dry_run: bool = Query(False, description="Validate only; create nothing"),
) -> ImportReport:
    content = await _read_upload(file)
    return service.run(
        filename=file.filename or "upload", content=content, dry_run=dry_run
    )


@router.post(
    "/cameras/import/validate",
    response_model=ImportReport,
    summary="Validate an import file without importing",
    responses={413: {"description": "File too large"}, 422: {"description": "Empty file"}},
    description=(
        "Identical to `POST /cameras/import?dry_run=true`, as a separate "
        "endpoint so a 'check my file' button cannot become an import by "
        "dropping one query parameter.\n\n"
        "Creates no cameras. The attempt itself is recorded, so the report "
        "stays retrievable by `job_id` and an operator can show what they "
        "checked before committing."
    ),
)
async def validate_import(
    service: ImportDep,
    file: UploadFile = File(description="A .csv, .xlsx or .json file of cameras"),
) -> ImportReport:
    content = await _read_upload(file)
    return service.run(
        filename=file.filename or "upload", content=content, dry_run=True
    )


@router.get(
    "/cameras/import/{job_id}",
    response_model=ImportReport,
    summary="Get a past import report",
    responses={404: {"description": "No import job with that id"}},
    description=(
        "The full report for an earlier import or validation, error list "
        "included -- so a 250-row upload that rejected nine rows can be "
        "worked through after the fact instead of being re-uploaded to see "
        "what went wrong."
    ),
)
def get_import_report(
    service: ImportDep,
    job_id: uuid.UUID = Path(description="The import job's UUID"),
) -> ImportReport:
    try:
        return service.get_report(job_id)
    except ImportJobNotFoundError as exc:
        raise _error(status.HTTP_404_NOT_FOUND, exc.code, exc.message) from exc


# ── official Sentinel portal ──────────────────────────────────────────────
#
# No client for the live portal exists: its API is not documented to us, and
# the alternative (reading its web frontend for internal calls) is scraping.
# Until it is, these endpoints run against UnconfiguredPortalClient and answer
# 503 with the exact checklist of what is missing -- the gap is visible from
# the running system, not hidden in a TODO.


def get_portal_session() -> ResilientPortalSession:
    """The portal session. Overridable in tests via dependency_overrides."""
    try:
        load_credentials()
        missing: list[str] = []
    except PortalNotConfiguredError as exc:
        missing = exc.missing
    return ResilientPortalSession(UnconfiguredPortalClient(missing))


PortalDep = Annotated[ResilientPortalSession, Depends(get_portal_session)]

#: Portal failure -> HTTP status. Upstream auth failures are 502, not 401:
#: the portal rejected THIS SERVER's credentials, which the API caller can do
#: nothing about, and a 401 would tell their client to re-prompt its own user.
_PORTAL_STATUS: list[tuple[type[PortalError], int]] = [
    (PortalNotConfiguredError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (AuthExpiredError, status.HTTP_502_BAD_GATEWAY),
    (AuthenticationError, status.HTTP_502_BAD_GATEWAY),
    (PermissionDeniedError, status.HTTP_403_FORBIDDEN),
    (CameraUnavailableError, status.HTTP_404_NOT_FOUND),
    (StreamOfflineError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (NoSupportedStreamError, status.HTTP_422_UNPROCESSABLE_ENTITY),
    (PortalTimeoutError, status.HTTP_504_GATEWAY_TIMEOUT),
]


def _portal_error(exc: PortalError) -> HTTPException:
    code = next(
        (http for kind, http in _PORTAL_STATUS if isinstance(exc, kind)),
        status.HTTP_502_BAD_GATEWAY,
    )
    detail: dict[str, Any] = {"code": exc.code, "message": exc.message}
    if isinstance(exc, PortalNotConfiguredError):
        detail["missing"] = exc.missing
    _logger.warning("Sentinel portal request failed: [%s] %s", exc.code, exc.message)
    return HTTPException(status_code=code, detail=detail)


class PortalStatusResponse(BaseModel):
    credentials_configured: bool
    client_available: bool = Field(
        description="False until a client exists for the documented portal API"
    )
    missing_settings: list[str]
    required_documentation: list[str]


class PortalCameraOut(BaseModel):
    portal_id: str
    name: str
    department: str | None
    district: str | None
    location: str | None
    latitude: float | None
    longitude: float | None
    status: str | None
    online: bool | None
    stream_types: list[str] = Field(description="Every type offered")
    availability: dict[str, bool] = Field(description="Type -> offered AND up now")


class StreamSelection(BaseModel):
    portal_id: str
    stream_type: str
    url: str
    priority: list[str] = Field(description="The order streams are chosen in")


class SyncResponse(BaseModel):
    discovered: int
    created: int
    updated: int
    unchanged: int
    skipped: list[dict[str, str]]
    errors: list[dict[str, str]]


@router.get(
    "/portal/status",
    response_model=PortalStatusResponse,
    summary="Sentinel portal integration status",
    tags=["sentinel-portal"],
    description=(
        "Reports whether credentials are configured and whether a client for "
        "the official portal API exists yet, and lists exactly what is still "
        "needed. Never returns credential values."
    ),
)
def portal_status() -> PortalStatusResponse:
    try:
        load_credentials()
        missing: list[str] = []
    except PortalNotConfiguredError as exc:
        missing = exc.missing
    return PortalStatusResponse(
        credentials_configured=not missing,
        client_available=False,
        missing_settings=missing,
        required_documentation=list(REQUIRED_INFORMATION),
    )


@router.get(
    "/portal/cameras",
    response_model=list[PortalCameraOut],
    summary="Discover cameras on the Sentinel portal",
    tags=["sentinel-portal"],
    responses={502: {"description": "Portal rejected credentials"},
               503: {"description": "Integration not configured"},
               504: {"description": "Portal timed out"}},
    description=(
        "Every camera visible to the configured account, with the stream "
        "types it offers and which are currently up. Read-only: nothing is "
        "written to the registry -- see `POST /portal/sync` for that."
    ),
)
def discover_portal_cameras(portal: PortalDep) -> list[PortalCameraOut]:
    try:
        catalogue = portal.list_cameras()
    except PortalError as exc:
        raise _portal_error(exc) from exc
    return [
        PortalCameraOut(
            portal_id=c.portal_id, name=c.name, department=c.department,
            district=c.district, location=c.location,
            latitude=c.latitude, longitude=c.longitude,
            status=c.status, online=c.online,
            stream_types=sorted({o.stream_type.value for o in c.streams}),
            availability=c.availability(),
        )
        for c in catalogue
    ]


@router.get(
    "/portal/cameras/{portal_id}/stream",
    response_model=StreamSelection,
    summary="Choose the stream to open for a portal camera",
    tags=["sentinel-portal"],
    responses={403: {"description": "Account lacks permission"},
               404: {"description": "Camera not in this account's catalogue"},
               422: {"description": "No RTSP/WHEP/HLS stream offered"},
               503: {"description": "Streams offered but all offline, or not configured"}},
    description=(
        "Picks the best currently-available stream by priority "
        "**RTSP > WHEP > HLS**. An offline camera (503) and one offering no "
        "playable type (422) are reported differently, because only the first "
        "is worth retrying."
    ),
)
def select_portal_stream(
    portal: PortalDep,
    portal_id: str = Path(description="The camera's id in the portal"),
) -> StreamSelection:
    try:
        camera = portal.get_camera(portal_id)
        offer = select_stream(camera)
    except PortalError as exc:
        raise _portal_error(exc) from exc
    _logger.info("Portal stream selected for %s: %s %s",
                 portal_id, offer.stream_type, mask_credentials(offer.url))
    from sentinel_system.portal import STREAM_PRIORITY

    return StreamSelection(
        portal_id=portal_id,
        stream_type=offer.stream_type.value,
        url=offer.url,
        priority=[t.value for t in STREAM_PRIORITY],
    )


@router.post(
    "/portal/sync",
    response_model=SyncResponse,
    summary="Populate the camera registry from the Sentinel portal",
    tags=["sentinel-portal"],
    responses={502: {"description": "Portal rejected credentials"},
               503: {"description": "Integration not configured"}},
    description=(
        "Discovers the catalogue and registers each camera through the same "
        "service as `POST /cameras` -- there is still one way into the "
        "registry. Portal cameras get the code `SEN-<portal id>` so they can "
        "never overwrite a camera registered by hand.\n\n"
        "Idempotent. A camera the portal describes without a district or "
        "coordinates is **skipped with the reason**, not given a made-up "
        "location. A re-sync never changes `status`, so a camera an operator "
        "decommissioned stays decommissioned."
    ),
)
def sync_portal_catalogue(
    portal: PortalDep,
    session: Annotated[Session, Depends(get_registry_session)],
) -> SyncResponse:
    try:
        report = CatalogueSync(session, portal).run()
    except PortalError as exc:
        raise _portal_error(exc) from exc
    return SyncResponse(**report.as_dict())


__all__ = [
    "router",
    "get_registry_session",
    "get_camera_service",
    "get_verification_service",
    "get_import_service",
    "get_portal_session",
]
