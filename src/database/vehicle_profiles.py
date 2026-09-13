"""
Vehicle profiles: one summary row per plate, derived from vehicle_events.

vehicle_events remains the source of truth -- every detection, unchanged.
A profile is a running summary of a plate's detections: its most likely
class and colour, its best images, where and when it was first and last
seen, and the sequence of camera visits.

Profiles are maintained two ways, through ONE function (_apply_event):

  * incrementally: insert_event() folds each new detection into the plate's
    profile in the same transaction (one indexed lookup and one update), so
    profiles are always current and cost almost nothing per vehicle;
  * by rebuild: replaying a plate's events oldest-first. Used when history
    changes underneath a profile (a session deleted, a detection arriving
    out of order) and to backfill an existing database the first time this
    table appears.

Because both paths run the same fold, an incrementally maintained profile
and a rebuilt one are identical -- which the tests assert.

This is deliberately independent of trajectory reconstruction
(src/trajectory): it records the order a vehicle visited cameras, it does
not reconstruct or measure a path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.database.models import VehicleEvent, VehicleProfile
from src.utils.logger import get_logger

_logger = get_logger("database.vehicle_profiles")

# Two detections of one plate at the same camera, in the same session, no
# further apart than this, are one visit rather than two. Matches the
# trajectory engine's default so the two agree on what a visit is.
DEFAULT_REVISIT_GAP_SECONDS = 300.0

# A profile keeps at most this many visits in its history (the newest). The
# events table still has everything; this only bounds one row's size for a
# vehicle seen thousands of times.
MAX_HISTORY_ENTRIES = 500

_VOTED_ATTRIBUTES = ("vehicle_class", "vehicle_color", "vehicle_type", "plate_color")
_UNINFORMATIVE = {None, "", "Unknown", "unknown"}


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(text: Optional[str], default):
    try:
        value = json.loads(text) if text else default
    except ValueError:
        return default
    return value if isinstance(value, type(default)) else default


# ── the fold ─────────────────────────────────────────────────────────────


def _apply_event(
    profile: VehicleProfile,
    event: Any,
    revisit_gap_seconds: float = DEFAULT_REVISIT_GAP_SECONDS,
) -> None:
    """Fold one detection into a profile. Events must arrive oldest-first."""
    timestamp = event.timestamp or _now_iso()

    # Times.
    if not profile.first_seen:
        profile.first_seen = timestamp
    profile.last_seen = timestamp
    profile.total_detections = (profile.total_detections or 0) + 1

    # Attributes by majority vote, ignoring "Unknown" so a crop the colour
    # detector could not read never outvotes the ones it could.
    votes = _load_json(profile.attribute_votes, {})
    for attribute in _VOTED_ATTRIBUTES:
        value = getattr(event, attribute, None)
        if value in _UNINFORMATIVE:
            continue
        tally = votes.setdefault(attribute, {})
        tally[value] = tally.get(value, 0) + 1
    for attribute in _VOTED_ATTRIBUTES:
        tally = votes.get(attribute) or {}
        if tally:
            # Ties go to the most recent value seen, which dict order gives
            # when counts are equal and the newer key was inserted later --
            # so resolve explicitly: highest count, then latest.
            best = max(tally.items(), key=lambda item: item[1])[1]
            leaders = [name for name, count in tally.items() if count == best]
            current = getattr(event, attribute, None)
            setattr(profile, attribute, current if current in leaders else leaders[-1])
    profile.attribute_votes = json.dumps(votes, sort_keys=True)

    # Last known location and session: the latest detection.
    profile.camera_id = event.camera_id
    profile.camera_name = event.camera_name
    profile.latitude = event.latitude
    profile.longitude = event.longitude
    profile.processing_session = event.processing_session
    profile.ocr_confidence = event.confidence

    # Images: from the highest-confidence detection that has them. A newer
    # detection with equal confidence wins, so the images stay recent.
    confidence = event.confidence
    has_images = bool(event.vehicle_thumbnail_path or event.plate_thumbnail_path
                      or event.vehicle_image_path or event.image_path)
    is_best = confidence is not None and (
        profile.best_confidence is None or confidence >= profile.best_confidence
    )
    if is_best:
        profile.best_confidence = confidence
    missing_images = not (profile.vehicle_thumbnail_path or profile.plate_thumbnail_path
                          or profile.vehicle_image_path or profile.plate_image_path)
    if has_images and (is_best or missing_images):
        profile.vehicle_image_path = event.vehicle_image_path or profile.vehicle_image_path
        profile.plate_image_path = event.image_path or profile.plate_image_path
        profile.vehicle_thumbnail_path = (
            event.vehicle_thumbnail_path or profile.vehicle_thumbnail_path
        )
        profile.plate_thumbnail_path = (
            event.plate_thumbnail_path or profile.plate_thumbnail_path
        )

    # Camera visits.
    history: list[dict] = _load_json(profile.trajectory_history, [])
    last = history[-1] if history else None
    event_time = _parse(timestamp)
    same_visit = False
    if last is not None and last.get("camera_id") == event.camera_id \
            and last.get("processing_session") == event.processing_session:
        previous = _parse(last.get("last_seen"))
        if previous is None or event_time is None:
            same_visit = True
        else:
            same_visit = (event_time - previous).total_seconds() <= revisit_gap_seconds
    if same_visit:
        last["last_seen"] = timestamp
        last["detections"] = int(last.get("detections", 1)) + 1
        if confidence is not None:
            last["confidence"] = max(confidence, last.get("confidence") or 0.0)
    else:
        history.append({
            "camera_id": event.camera_id,
            "camera_name": event.camera_name,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "processing_session": event.processing_session,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "detections": 1,
            "confidence": confidence,
        })
    if len(history) > MAX_HISTORY_ENTRIES:
        history = history[-MAX_HISTORY_ENTRIES:]

    profile.total_camera_visits = len(history)
    profile.unique_cameras = len({v.get("camera_id") for v in history if v.get("camera_id")})
    profile.trajectory_history = json.dumps(history)
    profile.updated_at = _now_iso()


def _new_profile(plate_number: str) -> VehicleProfile:
    now = _now_iso()
    return VehicleProfile(
        plate_number=plate_number,
        first_seen="",
        last_seen="",
        total_detections=0,
        total_camera_visits=0,
        unique_cameras=0,
        trajectory_history="[]",
        attribute_votes="{}",
        updated_at=now,
    )


# ── public API ───────────────────────────────────────────────────────────


def update_profile_for_event(
    session: Session,
    event: VehicleEvent,
    revisit_gap_seconds: float = DEFAULT_REVISIT_GAP_SECONDS,
) -> Optional[VehicleProfile]:
    """Fold a just-inserted detection into its plate's profile.

    Runs inside a SAVEPOINT, and never raises: the detection has already been
    written, and a profile problem -- including two processes creating the
    same new profile at once -- must never roll that back. On any failure
    the savepoint is discarded and the profile is simply rebuilt from events
    next time it is touched.
    """
    plate = (event.plate_number or "").strip().upper()
    if not plate:
        return None

    try:
        with session.begin_nested():
            profile = (
                session.query(VehicleProfile)
                .filter(VehicleProfile.plate_number == plate)
                .one_or_none()
            )
            if profile is not None and profile.last_seen:
                previous, current = _parse(profile.last_seen), _parse(event.timestamp)
                if previous and current and current < previous:
                    # Arrived out of order (a second process, a delayed API
                    # post). Appending would scramble the visit order, so
                    # replay this plate's history in time order instead.
                    return rebuild_profile(session, plate, revisit_gap_seconds)
            if profile is None:
                profile = _new_profile(plate)
                session.add(profile)
            _apply_event(profile, event, revisit_gap_seconds)
            session.flush()
            return profile
    except IntegrityError:
        # Another process created this plate's profile between our lookup
        # and our insert. The event is safe; rebuild picks up both.
        _logger.debug("Profile for %s created concurrently; rebuilding", plate)
        try:
            with session.begin_nested():
                return rebuild_profile(session, plate, revisit_gap_seconds)
        except Exception as exc:
            _logger.warning("Could not rebuild the profile for %s: %s", plate, exc)
            return None
    except Exception as exc:
        _logger.warning("Could not update the vehicle profile for %s: %s", plate, exc)
        return None


def rebuild_profile(
    session: Session,
    plate_number: str,
    revisit_gap_seconds: float = DEFAULT_REVISIT_GAP_SECONDS,
) -> Optional[VehicleProfile]:
    """Recompute a plate's profile from all of its stored detections.

    Deletes the profile if the plate has no detections left.
    """
    plate = (plate_number or "").strip().upper()
    # Case-insensitive on purpose. Profiles are keyed by the upper-cased
    # plate, but insert_event() stores a plate exactly as it was given, so a
    # lowercase plate posted to /entry would match nothing here -- and an
    # exact match then DELETED that plate's profile as "no events left".
    events: Iterable[VehicleEvent] = (
        session.query(VehicleEvent)
        .filter(func.upper(VehicleEvent.plate_number) == plate)
        .order_by(VehicleEvent.timestamp.asc(), VehicleEvent.id.asc())
        .all()
    )
    events = list(events)
    profile = (
        session.query(VehicleProfile)
        .filter(VehicleProfile.plate_number == plate)
        .one_or_none()
    )
    if not events:
        if profile is not None:
            session.delete(profile)
            session.flush()
        return None

    fresh = _new_profile(plate)
    for event in events:
        _apply_event(fresh, event, revisit_gap_seconds)

    if profile is None:
        session.add(fresh)
        profile = fresh
    else:
        for column in VehicleProfile.__table__.columns:
            if column.name not in ("id", "plate_number"):
                setattr(profile, column.name, getattr(fresh, column.name))
    session.flush()
    return profile


def rebuild_profiles(
    session: Session,
    plate_numbers: Iterable[str],
    revisit_gap_seconds: float = DEFAULT_REVISIT_GAP_SECONDS,
) -> int:
    """Rebuild several plates' profiles. Returns how many were processed."""
    count = 0
    for plate in {p.strip().upper() for p in plate_numbers if p}:
        rebuild_profile(session, plate, revisit_gap_seconds)
        count += 1
    return count


def rebuild_all_profiles(
    session: Session,
    revisit_gap_seconds: float = DEFAULT_REVISIT_GAP_SECONDS,
) -> int:
    """Rebuild every profile from vehicle_events. Returns the plate count."""
    plates = [row[0] for row in session.query(func.upper(VehicleEvent.plate_number)).distinct()]
    session.query(VehicleProfile).delete(synchronize_session=False)
    session.flush()
    return rebuild_profiles(session, plates, revisit_gap_seconds)


def get_profile(session: Session, plate_number: str) -> Optional[dict]:
    """A plate's profile as a dict, or None if the plate is unknown."""
    profile = (
        session.query(VehicleProfile)
        .filter(VehicleProfile.plate_number == (plate_number or "").strip().upper())
        .one_or_none()
    )
    return profile.to_dict() if profile is not None else None


def list_profiles(session: Session, limit: int = 100) -> list[dict]:
    """Most recently seen vehicles first."""
    rows = (
        session.query(VehicleProfile)
        .order_by(VehicleProfile.last_seen.desc())
        .limit(limit)
        .all()
    )
    return [row.to_dict() for row in rows]


def search_profiles(session: Session, query, limit: int = 60) -> list[dict]:
    """Profiles matching a parsed natural-language query (search.nl_query).

    Colour, class and plate are filtered in SQL. Time and camera are checked
    per camera visit, so "white cars at India Gate after 9am" needs ONE visit
    that is both at India Gate and after 9am -- not a vehicle seen at India
    Gate once and somewhere else after 9am. Each result carries
    `matched_visit`, the most recent visit that satisfied the query.
    """
    rows = session.query(VehicleProfile)
    if query.colors:
        rows = rows.filter(VehicleProfile.vehicle_color.in_(query.colors))
    if query.classes:
        rows = rows.filter(VehicleProfile.vehicle_class.in_(query.classes))
    if query.plate_fragment:
        rows = rows.filter(VehicleProfile.plate_number.contains(query.plate_fragment))

    results: list[dict] = []
    for row in rows.order_by(VehicleProfile.last_seen.desc()).all():
        profile = row.to_dict()
        matched = None
        for visit in reversed(profile["trajectory_history"]):
            if query.camera_ids and visit.get("camera_id") not in query.camera_ids:
                continue
            first, last = _parse(visit.get("first_seen")), _parse(visit.get("last_seen"))
            if query.start and (last is None or last < query.start):
                continue
            if query.end and (first is None or first > query.end):
                continue
            matched = visit
            break
        if matched is None and (query.camera_ids or query.start or query.end):
            continue
        profile["matched_visit"] = matched or (profile["trajectory_history"] or [None])[-1]
        results.append(profile)
        if len(results) >= limit:
            break
    return results
