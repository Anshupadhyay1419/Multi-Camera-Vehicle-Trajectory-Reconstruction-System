"""Camera verification: prove a registered camera is actually usable.

Explicitly operator-triggered and bounded. This is NOT health monitoring --
there is no scheduler, no background worker and no daemon here; it runs
once, when asked, and closes everything it opened.

All RTSP work is delegated to `src.cameras.stream_probe`, the module the
ALPR system already uses for bounded stream checks. Nothing here opens a
socket or touches OpenCV directly: a second RTSP implementation would be
one more thing to keep correct about timeouts, thread abandonment and
credential masking, and those are already solved there.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.cameras.stream_probe import StreamSample, mask_credentials, sample_stream
from src.utils.logger import get_logger

from sentinel_system.core.config import Settings, get_settings
from sentinel_system.registry.exceptions import CameraNotFoundError
from sentinel_system.registry.models import Camera
from sentinel_system.registry.repository import CameraRepository
from sentinel_system.verification.enums import ConnectionStatus
from sentinel_system.verification.models import CameraVerification
from sentinel_system.verification.schemas import VerificationResult

_logger = get_logger("sentinel.verification")

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Fragments OpenCV / the probe use for each failure, mapped to the status an
#: operator can act on. Ordered: the first match wins, so the specific
#: phrases come before the general ones.
_STATUS_HINTS: tuple[tuple[str, ConnectionStatus], ...] = (
    ("did not finish within", ConnectionStatus.TIMEOUT),
    ("no response from", ConnectionStatus.TIMEOUT),
    ("no video arrived", ConnectionStatus.TIMEOUT),
    ("cannot resolve", ConnectionStatus.UNREACHABLE),
    ("refused the connection", ConnectionStatus.UNREACHABLE),
    ("cannot reach", ConnectionStatus.UNREACHABLE),
    ("sent no frames", ConnectionStatus.NO_FRAMES),
    ("would not open", ConnectionStatus.STREAM_ERROR),
)


def _classify(sample: StreamSample) -> ConnectionStatus:
    """Turn a sample's message into the status an operator acts on.

    Matched on text because OpenCV does not expose a failure code -- it
    reports "could not open" whether the path is wrong, the password is
    wrong or the codec is unsupported. Rather than invent a precision the
    library does not have, the coarse status is paired with the exact
    message, and STREAM_ERROR is documented as "check path and credentials"
    for that reason.
    """
    if sample.ok:
        return ConnectionStatus.REACHABLE
    message = (sample.message or "").lower()
    for fragment, status in _STATUS_HINTS:
        if fragment in message:
            return status
    return ConnectionStatus.STREAM_ERROR


class CameraVerificationService:
    """Verify one camera, record the result, and return it."""

    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        repository: CameraRepository | None = None,
        sampler: Any = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.repository = repository or CameraRepository(session)
        # Injected so tests can drive every branch -- timeout, auth failure,
        # no frames -- without a camera or a network.
        self._sample = sampler or sample_stream

    # ── the use-case ──────────────────────────────────────────────────────

    def verify(self, camera_id: uuid.UUID) -> VerificationResult:
        """Open the camera's stream, measure it, store what was found.

        Never raises for a camera that simply does not work: an unreachable
        camera is a recorded result with `verified=false`, not an exception.
        The one genuine error is a camera that is not registered at all.

        Raises:
            CameraNotFoundError: no camera with that id.
        """
        camera = self.repository.get_by_id(camera_id)
        if camera is None:
            raise CameraNotFoundError(camera_id=camera_id)

        safe_url = mask_credentials(camera.stream_url)
        _logger.info(
            "Verification started: %s (%s) stream=%s",
            camera.camera_code, camera.id, safe_url,
        )

        sample = self._run_sample(camera)
        status = _classify(sample)
        errors = [] if sample.ok else [sample.message]

        thumbnail = None
        if sample.ok and sample.frame is not None:
            thumbnail, thumbnail_error = self._save_thumbnail(camera, sample.frame)
            if thumbnail_error:
                # A thumbnail that could not be written does not invalidate
                # a stream that demonstrably delivered frames; it is
                # reported alongside the successful measurement.
                errors.append(thumbnail_error)

        record = CameraVerification(
            camera_id=camera.id,
            verified=bool(sample.ok),
            connection_status=status,
            verification_message=sample.message,
            measured_resolution=sample.resolution,
            measured_fps=sample.measured_fps,
            measured_latency_ms=self._headline_latency(sample),
            connect_latency_ms=sample.connect_latency_ms,
            first_frame_latency_ms=sample.first_frame_latency_ms,
            frames_sampled=sample.frames_read,
            codec_detected=sample.fourcc,
            thumbnail_path=thumbnail,
            errors=errors,
            checked_at=datetime.now(timezone.utc),
        )
        self.session.add(record)
        self.session.commit()

        if record.verified:
            _logger.info(
                "Verification completed: %s (%s) %s @ %sfps latency=%sms codec=%s",
                camera.camera_code, camera.id, record.measured_resolution,
                record.measured_fps, record.measured_latency_ms,
                record.codec_detected or "unknown",
            )
        else:
            _logger.warning(
                "Verification failed: %s (%s) status=%s stream=%s: %s",
                camera.camera_code, camera.id, record.connection_status,
                safe_url, record.verification_message,
            )
        return VerificationResult.model_validate(record)

    # ── reads ─────────────────────────────────────────────────────────────

    def latest(self, camera_id: uuid.UUID) -> VerificationResult | None:
        """The most recent verification of a camera, if it has ever had one."""
        statement = (
            select(CameraVerification)
            .where(CameraVerification.camera_id == camera_id)
            .order_by(
                CameraVerification.checked_at.desc(), CameraVerification.id.desc()
            )
            .limit(1)
        )
        record = self.session.execute(statement).scalar_one_or_none()
        return None if record is None else VerificationResult.model_validate(record)

    def history(self, camera_id: uuid.UUID, limit: int = 20) -> list[VerificationResult]:
        """Past verifications, newest first."""
        statement = (
            select(CameraVerification)
            .where(CameraVerification.camera_id == camera_id)
            .order_by(
                CameraVerification.checked_at.desc(), CameraVerification.id.desc()
            )
            .limit(limit)
        )
        rows = self.session.execute(statement).scalars().all()
        return [VerificationResult.model_validate(row) for row in rows]

    # ── internals ─────────────────────────────────────────────────────────

    def _run_sample(self, camera: Camera) -> StreamSample:
        """Read the stream, converting any unexpected fault into a result.

        sample_stream() already promises not to raise. This is the belt to
        that braces: the endpoint's contract is that a broken camera never
        produces a 500, and a bug in the sampler must not be the exception
        to it.
        """
        try:
            return self._sample(
                camera.stream_url,
                frames=self.settings.frames_to_sample,
                timeout=self.settings.verification_timeout,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _logger.exception("Verification sampler failed unexpectedly")
            return StreamSample(ok=False, message=f"Verification failed: {exc}")

    @staticmethod
    def _headline_latency(sample: StreamSample) -> float | None:
        """Connect + open + first frame: what the operator actually waited.

        Summed rather than reporting one of them, because each alone is
        misleading -- a fast TCP connect to a camera that then takes four
        seconds to produce a picture is not a fast camera.
        """
        parts = [
            sample.connect_latency_ms,
            sample.open_latency_ms,
            sample.first_frame_latency_ms,
        ]
        present = [value for value in parts if value is not None]
        return round(sum(present), 1) if present else None

    def _save_thumbnail(self, camera: Camera, frame: Any) -> tuple[str | None, str | None]:
        """Write one JPEG and return (served path, error).

        Written under the ALPR API's existing `data/thumbnails` tree, which
        is already mounted as a static route -- so the returned path is
        directly loadable by a browser without adding a route or an
        endpoint that streams bytes back.

        Named by camera id, not by timestamp: the useful thumbnail is the
        latest one, and keeping one file per camera stops an operator who
        clicks Verify repeatedly from filling the disk. The history table
        still records that each attempt produced one.
        """
        try:
            import cv2

            directory = Path(self.settings.thumbnail_directory)
            if not directory.is_absolute():
                directory = _REPO_ROOT / directory
            directory.mkdir(parents=True, exist_ok=True)

            filename = f"{camera.id}.jpg"
            written = cv2.imwrite(
                str(directory / filename),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.thumbnail_quality],
            )
            if not written:
                return None, "The thumbnail could not be written."
            return f"/thumbnails/cameras/{filename}", None
        except Exception as exc:
            _logger.warning("Thumbnail for %s could not be saved: %s", camera.id, exc)
            return None, f"The thumbnail could not be saved: {exc}"


__all__ = ["CameraVerificationService"]
