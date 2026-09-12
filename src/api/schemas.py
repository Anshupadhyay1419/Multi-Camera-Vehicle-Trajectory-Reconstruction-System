"""
Pydantic request/response schemas for the ALPR FastAPI server.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class EntryRequest(BaseModel):
    plate_number: str
    vehicle_type: str
    plate_color:  str
    series_type:  str
    direction:    Literal["IN", "OUT"]
    image_path:   str = ""

    # Which camera saw this vehicle, and where that camera is. Optional: a
    # caller on the capturing device can leave these out and the server
    # stamps its own config.yaml `camera:` values instead (see
    # server.create_entry), which is the normal single-gate case. Send them
    # explicitly only when posting on behalf of a *different* camera than
    # the one this API is running next to.
    camera_id:    Optional[str]   = None
    camera_name:  Optional[str]   = None
    latitude:     Optional[float] = None
    longitude:    Optional[float] = None


class EventResponse(BaseModel):
    id:           int
    plate_number: str
    vehicle_type: str
    plate_color:  str
    series_type:  str
    timestamp:    str
    direction:    str
    image_path:   str

    # Nullable in the database, so nullable here: events recorded before
    # camera attribution existed have none, and a gate that hasn't been
    # surveyed has no coordinates. The dashboard renders missing values as
    # a dash rather than hiding the column.
    camera_id:    Optional[str]   = None
    camera_name:  Optional[str]   = None
    latitude:     Optional[float] = None
    longitude:    Optional[float] = None
