"""
GeoJSON (RFC 7946) serialisation for reconstructed trajectories.

Produces a FeatureCollection holding:

  * one LineString  -- the path itself, for a polyline layer
  * one Point per camera visit -- for markers, with every popup field
    (camera name, plate, time, confidence, thumbnails) carried in
    `properties` so a map client needs no second request

Coordinates are [longitude, latitude] -- GeoJSON's axis order, the reverse
of the [lat, lon] pairs Leaflet's own API takes. Getting this backwards
plots Delhi in the Indian Ocean, so the conversion happens exactly once,
here, and the map layer consumes what this produces.

Points with no coordinates (an unsurveyed camera) are omitted from the
geometry rather than emitted at [0, 0]: null island is a worse answer than
a shorter line. They remain in the Trajectory itself, so the timeline and
history table still list them.
"""

from __future__ import annotations

from typing import Any, Optional

from src.trajectory.models import Trajectory, TrajectoryPoint


def _point_properties(point: TrajectoryPoint, plate_number: str) -> dict[str, Any]:
    """Everything a map popup shows for one camera visit."""
    return {
        "sequence": point.sequence,
        "plate_number": plate_number,
        "camera_id": point.camera_id,
        "camera_name": point.camera_name,
        "timestamp": point.timestamp,
        "confidence": point.confidence,
        "detection_count": point.detection_count,
        "vehicle_type": point.vehicle_type,
        "plate_color": point.plate_color,
        "direction": point.direction,
        "plate_image_path": point.plate_image_path,
        "vehicle_image_path": point.vehicle_image_path,
        "video_source": point.video_source,
        "ocr_text": point.ocr_text,
    }


def trajectory_to_geojson(trajectory: Trajectory) -> dict[str, Any]:
    """Render *trajectory* as a GeoJSON FeatureCollection.

    Args:
        trajectory: The reconstructed path.

    Returns:
        A FeatureCollection dict, JSON-serialisable as-is. Always valid,
        including for an empty trajectory (an empty `features` list) and for
        one whose cameras have no coordinates -- a client should never have
        to special-case the shape of this response.
    """
    mappable = trajectory.mappable_points
    features: list[dict[str, Any]] = []

    # LineString first so a client drawing features in order puts the path
    # underneath its markers. Needs at least two positions to be a valid
    # LineString; a single sighting is a point, not a path.
    if len(mappable) >= 2:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [point.longitude, point.latitude] for point in mappable
                ],
            },
            "properties": {
                "kind": "path",
                "plate_number": trajectory.plate_number,
                "cameras_visited": trajectory.cameras_visited,
                "total_distance_km": trajectory.total_distance_km,
                "duration_seconds": trajectory.duration_seconds,
                "first_seen": trajectory.first_seen,
                "last_seen": trajectory.last_seen,
                "ordering": trajectory.ordering,
            },
        })

    for point in mappable:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [point.longitude, point.latitude],
            },
            "properties": {"kind": "detection", **_point_properties(point, trajectory.plate_number)},
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        # Summary carried alongside the features so a client can render
        # headline figures without walking the geometry. Non-standard but
        # explicitly permitted: RFC 7946 allows foreign members.
        "properties": {
            "plate_number": trajectory.plate_number,
            "processing_session": trajectory.processing_session,
            "ordering": trajectory.ordering,
            "cameras_visited": trajectory.cameras_visited,
            "total_detections": trajectory.total_detections,
            "mapped_points": len(mappable),
            "unmapped_points": len(trajectory.points) - len(mappable),
            "total_distance_km": trajectory.total_distance_km,
            "duration_seconds": trajectory.duration_seconds,
            "average_confidence": trajectory.average_confidence,
            "path_labels": trajectory.path_labels,
        },
    }


def ordered_coordinates(trajectory: Trajectory) -> list[list[float]]:
    """The path as [latitude, longitude] pairs, in travel order.

    Leaflet's axis order, not GeoJSON's -- this is what
    `L.polyline(...)` and `map.fitBounds(...)` take directly.
    """
    return [[point.latitude, point.longitude] for point in trajectory.mappable_points]


def ordered_locations(trajectory: Trajectory) -> list[dict[str, Any]]:
    """The path as an ordered list of visited locations.

    The timeline view's data source: every visit in order, including ones
    with no coordinates, each with the name, time and confidence a timeline
    row shows.
    """
    return [
        {
            "sequence": point.sequence,
            "camera_id": point.camera_id,
            "camera_name": point.camera_name,
            "latitude": point.latitude,
            "longitude": point.longitude,
            "timestamp": point.timestamp,
            "confidence": point.confidence,
            "detection_count": point.detection_count,
            "has_location": point.has_location,
        }
        for point in trajectory.points
    ]


def bounding_box(trajectory: Trajectory) -> Optional[list[float]]:
    """[min_lon, min_lat, max_lon, max_lat] for the mapped points, or None.

    GeoJSON `bbox` axis order. None when nothing is mappable, so a caller
    can tell "no extent" apart from "a zero-area extent at one camera".
    """
    mappable = trajectory.mappable_points
    if not mappable:
        return None
    lats = [point.latitude for point in mappable]
    lons = [point.longitude for point in mappable]
    return [min(lons), min(lats), max(lons), max(lats)]
