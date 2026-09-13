"""
Natural-language vehicle search: turn a sentence into filters, run them.

    "show all white cars"                    -> colour=White, class=car
    "red trucks after 9am today"             -> + time window
    "blue car at India Gate last 2 hours"    -> + camera
    "silver vehicles yesterday"              -> colour only, whole day

Understands every stored vehicle feature: body colour, vehicle type, plate,
plate colour, registration category, BH series, direction, cameras visited,
OCR confidence, time, and camera.

Parsed locally with rules, not by a language model: this device has no
model or API key configured, the vocabulary is small and fixed, and a
deterministic parser answers in
microseconds offline. Words it does not recognise are ignored rather than
guessed at, and the dashboard shows what was understood so the operator can
see why a result matched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Iterable, Optional

from src.classification.vehicle_color import SUPPORTED_COLORS

_COLOR_WORDS = {c.lower(): c for c in SUPPORTED_COLORS}
_COLOR_WORDS.update({"grey": "Gray", "maroon": "Red", "golden": "Yellow", "gold": "Yellow",
                     "navy": "Blue", "beige": "Brown", "cream": "White"})
_CLASS_WORDS = {"car": "car", "cars": "car", "sedan": "car", "suv": "car", "hatchback": "car",
                "truck": "truck", "trucks": "truck", "lorry": "truck", "van": "truck",
                "bus": "bus", "buses": "bus", "motorcycle": "motorcycle", "motorcycles": "motorcycle",
                "bike": "motorcycle", "bikes": "motorcycle", "scooter": "motorcycle"}
# A full Indian plate (DL8CA1234, KA02ACS4999, 22BH1234AA), or an explicit
# "plate <fragment>" for a partial one. Loose patterns matched ordinary text
# ("on 2026-09-12" read as plate ON2026).
_PLATE = re.compile(r"\b([A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}|\d{2}BH\d{4}[A-Z]{1,2})\b", re.I)
_PLATE_FRAGMENT = re.compile(r"\bplate\s+([A-Z0-9]{2,10})\b", re.I)
_CLOCK = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
# Registration category (stored as vehicle_type, inferred from plate colour).
_REGISTRATION_WORDS = {"private": "Private", "commercial": "Commercial", "taxi": "Commercial",
                       "cab": "Commercial", "ev": "EV", "electric": "EV",
                       "government": "Govt/Temp", "govt": "Govt/Temp", "temporary": "Govt/Temp",
                       "diplomatic": "Diplomatic", "rental": "Rental", "military": "Military",
                       "army": "Military"}
_PLATE_COLOURS = {"white": "White", "yellow": "Yellow", "green": "Green", "red": "Red",
                  "blue": "Blue", "black": "Black"}


@dataclass
class VehicleQuery:
    colors: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    start: Optional[datetime] = None          # timezone-aware
    end: Optional[datetime] = None
    camera_ids: list[str] = field(default_factory=list)
    plate_fragment: Optional[str] = None
    vehicle_types: list[str] = field(default_factory=list)   # registration category
    plate_colors: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)       # IN / OUT
    min_cameras: Optional[int] = None
    min_confidence: Optional[float] = None
    series: Optional[str] = None                              # "BH"

    @property
    def is_empty(self) -> bool:
        return not (self.colors or self.classes or self.start or self.end
                    or self.camera_ids or self.plate_fragment or self.vehicle_types
                    or self.plate_colors or self.directions or self.min_cameras
                    or self.min_confidence is not None or self.series)

    def describe(self, camera_names: Optional[dict] = None) -> str:
        parts = []
        if self.colors:
            parts.append("colour " + " or ".join(self.colors))
        if self.classes:
            parts.append("type " + " or ".join(self.classes))
        fmt = "%d %b %H:%M"
        if self.start and self.end:
            parts.append(f"seen {self.start:{fmt}} – {self.end:{fmt}}")
        elif self.start:
            parts.append(f"seen after {self.start:{fmt}}")
        elif self.end:
            parts.append(f"seen before {self.end:{fmt}}")
        if self.camera_ids:
            names = camera_names or {}
            parts.append("at " + " or ".join(names.get(c, c) for c in self.camera_ids))
        if self.plate_fragment:
            parts.append(f"plate containing {self.plate_fragment}")
        if self.vehicle_types:
            parts.append("registration " + " or ".join(self.vehicle_types))
        if self.plate_colors:
            parts.append(" or ".join(self.plate_colors) + " plate")
        if self.series:
            parts.append(f"{self.series} series")
        if self.directions:
            parts.append("direction " + " or ".join(self.directions))
        if self.min_cameras:
            parts.append(f"seen at {self.min_cameras}+ cameras")
        if self.min_confidence is not None:
            parts.append(f"OCR confidence ≥ {self.min_confidence:.0%}")
        return ", ".join(parts) if parts else "no filters (all vehicles)"


def _clock(hour: str, minute: Optional[str], meridiem: Optional[str]) -> Optional[time]:
    h, m = int(hour), int(minute or 0)
    if meridiem:
        if not 1 <= h <= 12:
            return None
        h = (h % 12) + (12 if meridiem.lower() == "pm" else 0)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return time(h, m)


def parse_query(text: str, now: Optional[datetime] = None,
                cameras: Iterable[tuple[str, str]] = ()) -> VehicleQuery:
    """Parse a search sentence. *cameras* is (camera_id, camera_name) pairs."""
    now = (now or datetime.now().astimezone())
    raw = (text or "").strip()
    lowered = raw.lower()
    words = re.findall(r"[a-z]+", lowered)
    query = VehicleQuery()

    def add(bucket: list, value) -> None:
        if value not in bucket:
            bucket.append(value)

    for index, word in enumerate(words):
        following = words[index + 1] if index + 1 < len(words) else ""
        # "yellow plate" is the PLATE's colour; "yellow car" is the body's.
        if following in ("plate", "plates", "number") and word in _PLATE_COLOURS:
            add(query.plate_colors, _PLATE_COLOURS[word])
            continue
        if word in _COLOR_WORDS:
            add(query.colors, _COLOR_WORDS[word])
        if word in _CLASS_WORDS:
            add(query.classes, _CLASS_WORDS[word])
        if word in _REGISTRATION_WORDS:
            add(query.vehicle_types, _REGISTRATION_WORDS[word])
        if word in ("entering", "entered", "entry", "incoming", "arriving"):
            add(query.directions, "IN")
        if word in ("exiting", "exited", "exit", "leaving", "outgoing", "departing"):
            add(query.directions, "OUT")

    if re.search(r"\bbh\b", lowered) and ("series" in words or "bharat" in words):
        query.series = "BH"

    cameras_count = re.search(
        r"\b(?:at least|min(?:imum)?|more than|over|>=?)\s*(\d+)\s*(?:\+\s*)?cameras?\b|\b(\d+)\s*\+\s*cameras?\b",
        lowered)
    if cameras_count:
        number = int(cameras_count.group(1) or cameras_count.group(2))
        query.min_cameras = number + 1 if "more than" in cameras_count.group(0) or "over" in cameras_count.group(0) else number
    elif re.search(r"\b(?:multiple|several|many|different)\s+cameras\b", lowered):
        query.min_cameras = 2

    confidence = re.search(r"\bconfidence\s*(?:above|over|>=?|at least)?\s*(\d+(?:\.\d+)?)\s*%?", lowered)
    if confidence:
        value = float(confidence.group(1))
        query.min_confidence = value / 100 if value > 1 else value
    elif re.search(r"\bhigh(?:ly)?\s+confiden", lowered):
        query.min_confidence = 0.9

    for camera_id, name in cameras:
        if name and name.lower() in lowered or camera_id.lower() in lowered:
            query.camera_ids.append(camera_id)

    plate = _PLATE.search(raw) or _PLATE_FRAGMENT.search(raw)
    if plate:
        query.plate_fragment = plate.group(1).upper()

    # ── day ──────────────────────────────────────────────────────────────
    day = None
    if "yesterday" in words:
        day = (now - timedelta(days=1)).date()
    elif "today" in words or "tonight" in words:
        day = now.date()
    date_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", lowered)
    if date_match:
        try:
            day = datetime(*map(int, date_match.groups())).date()
        except ValueError:
            pass
    tz = now.tzinfo

    def at(clock: time) -> datetime:
        return datetime.combine(day or now.date(), clock, tzinfo=tz)

    # ── relative window: last/past N minutes|hours|days ───────────────────
    relative = re.search(r"\b(?:last|past)\s+(\d+)?\s*(minute|min|hour|hr|day)s?\b", lowered)
    if relative:
        amount = int(relative.group(1) or 1)
        unit = relative.group(2)
        delta = (timedelta(minutes=amount) if unit.startswith("min")
                 else timedelta(hours=amount) if unit.startswith("h")
                 else timedelta(days=amount))
        query.start, query.end = now - delta, now
        return query

    # ── clock times ──────────────────────────────────────────────────────
    between = re.search(rf"\bbetween\s+{_CLOCK}\s+(?:and|to|-)\s+{_CLOCK}", lowered)
    if between:
        g = between.groups()
        first, second = _clock(g[0], g[1], g[2] or g[5]), _clock(g[3], g[4], g[5] or g[2])
        if first and second:
            query.start, query.end = at(first), at(second)
            return query
    after = re.search(rf"\b(?:after|since|from)\s+{_CLOCK}", lowered)
    before = re.search(rf"\b(?:before|until|till)\s+{_CLOCK}", lowered)
    if after and (clock := _clock(*after.groups())):
        query.start = at(clock)
    if before and (clock := _clock(*before.groups())):
        query.end = at(clock)
    if query.start or query.end:
        if day is not None:
            # "yesterday after 9am" means 9am until the END of yesterday.
            query.start = query.start or datetime.combine(day, time.min, tzinfo=tz)
            query.end = query.end or datetime.combine(day, time.max, tzinfo=tz)
        return query

    if day is not None:
        query.start = datetime.combine(day, time.min, tzinfo=tz)
        query.end = datetime.combine(day, time.max, tzinfo=tz)
    return query
