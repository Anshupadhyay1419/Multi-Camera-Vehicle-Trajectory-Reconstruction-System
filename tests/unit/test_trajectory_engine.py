"""
Unit tests for trajectory reconstruction (trajectory/engine.py, geojson.py).

The engine turns a bag of detections into an ordered path, so the tests
concentrate on the parts where getting it wrong produces a plausible-looking
but false answer:

- ordering, including the fallback when timestamps are unusable
- collapsing repeat reads at one camera WITHOUT deleting a genuine revisit
- measurements that must degrade to None rather than to a wrong number
- GeoJSON axis order, which is the reverse of Leaflet's
"""

from __future__ import annotations

import pytest

from src.trajectory import (
    ORDER_TIMESTAMP,
    ORDER_TRAJECTORY,
    TrajectoryEngine,
    bounding_box,
    haversine_km,
    initial_bearing_degrees,
    ordered_coordinates,
    ordered_locations,
    trajectory_to_geojson,
)

# The four demo sites, with their real coordinates.
CAMERAS = {
    1: ("CAM001", "India Gate", 28.6129, 77.2295),
    2: ("CAM002", "Connaught Place", 28.6315, 77.2167),
    3: ("CAM003", "Karol Bagh", 28.6519, 77.1909),
    4: ("CAM004", "Kashmere Gate", 28.6675, 77.2273),
}


def detection(order: int, minute: int, second: int = 0, confidence: float = 0.9, **overrides):
    camera_id, name, lat, lon = CAMERAS[order]
    base = {
        "camera_id": camera_id,
        "camera_name": name,
        "latitude": lat,
        "longitude": lon,
        "trajectory_order": order,
        "timestamp": f"2026-09-12T09:{minute:02d}:{second:02d}+00:00",
        "confidence": confidence,
        "image_path": f"plate_{order}.jpg",
        "vehicle_image_path": f"vehicle_{order}.jpg",
        "vehicle_type": "Car",
        "plate_color": "White",
        "direction": "IN",
        "processing_session": "SESS1",
        "video_source": f"{camera_id}.mp4",
        "ocr_text": "DL8CA1234",
    }
    base.update(overrides)
    return base


class TestOrdering:
    def test_reconstructs_the_path_in_travel_order(self):
        trajectory = TrajectoryEngine().build(
            "DL8CA1234", [detection(o, m) for o, m in zip([1, 2, 3, 4], [0, 12, 25, 48])]
        )
        assert trajectory.path_labels == [
            "India Gate", "Connaught Place", "Karol Bagh", "Kashmere Gate"
        ]
        assert [p.sequence for p in trajectory.points] == [1, 2, 3, 4]

    def test_input_order_does_not_matter(self):
        """Detections arrive newest-first, oldest-first or arbitrarily
        depending on the query; the reconstruction must not depend on it."""
        forward = [detection(o, m) for o, m in zip([1, 2, 3, 4], [0, 12, 25, 48])]
        engine = TrajectoryEngine()
        expected = engine.build("X", forward).path_labels
        assert engine.build("X", list(reversed(forward))).path_labels == expected
        assert engine.build("X", [forward[2], forward[0], forward[3], forward[1]]).path_labels == expected

    def test_auto_uses_timestamps_when_every_point_has_one(self):
        trajectory = TrajectoryEngine().build(
            "X", [detection(o, m) for o, m in zip([1, 2], [0, 10])]
        )
        assert trajectory.ordering == ORDER_TIMESTAMP

    def test_auto_falls_back_to_camera_order_without_timestamps(self):
        """A path ordered half by clock and half by config is not a sequence
        anyone can reason about, so the fallback is all-or-nothing."""
        detections = [detection(o, 0, timestamp=None) for o in (4, 1, 3, 2)]
        trajectory = TrajectoryEngine().build("X", detections)
        assert trajectory.ordering == ORDER_TRAJECTORY
        assert trajectory.path_labels == [
            "India Gate", "Connaught Place", "Karol Bagh", "Kashmere Gate"
        ]

    def test_one_unusable_timestamp_triggers_the_fallback(self):
        detections = [detection(1, 0), detection(2, 10, timestamp="not-a-date"), detection(3, 20)]
        assert TrajectoryEngine().build("X", detections).ordering == ORDER_TRAJECTORY

    def test_explicit_trajectory_order_overrides_timestamps(self):
        """Forcing camera order must win even when the clocks disagree --
        this is what makes a demo run deterministic."""
        detections = [detection(1, 50), detection(2, 10)]   # clock says 2 then 1
        trajectory = TrajectoryEngine(order_by=ORDER_TRAJECTORY).build("X", detections)
        assert trajectory.path_labels == ["India Gate", "Connaught Place"]

    def test_unknown_ordering_degrades_to_auto(self):
        engine = TrajectoryEngine(order_by="by-vibes")
        assert engine.order_by == "auto"


class TestCollapsing:
    def test_repeat_reads_at_one_camera_become_one_visit(self):
        detections = [detection(1, 0, s) for s in (0, 2, 5, 9)]
        trajectory = TrajectoryEngine().build("X", detections)
        assert len(trajectory.points) == 1
        assert trajectory.points[0].detection_count == 4
        assert trajectory.total_detections == 4

    def test_a_visit_reports_first_time_but_best_confidence(self):
        """The vehicle ARRIVED at the first read, but the clearest read is
        the one worth showing in a popup."""
        detections = [
            detection(1, 0, 0, confidence=0.61),
            detection(1, 0, 4, confidence=0.97, image_path="best.jpg"),
        ]
        point = TrajectoryEngine().build("X", detections).points[0]
        assert point.timestamp.endswith("09:00:00+00:00")
        assert point.confidence == pytest.approx(0.97)
        assert point.plate_image_path == "best.jpg"

    def test_a_genuine_revisit_is_not_collapsed_away(self):
        """A -> B -> A must stay three points. Grouping by camera_id alone
        would silently delete the return leg."""
        detections = [detection(1, 0), detection(2, 20), detection(1, 45)]
        trajectory = TrajectoryEngine(revisit_gap_seconds=300).build("X", detections)
        assert trajectory.path_labels == ["India Gate", "Connaught Place", "India Gate"]
        assert trajectory.cameras_visited == 2       # distinct cameras
        assert len(trajectory.points) == 3           # distinct visits

    def test_collapsing_can_be_disabled(self):
        detections = [detection(1, 0, s) for s in (0, 2, 5)]
        trajectory = TrajectoryEngine(collapse_per_camera=False).build("X", detections)
        assert len(trajectory.points) == 3

    def test_revisit_gap_boundary(self):
        """Inside the gap is one visit; beyond it is two."""
        near = [detection(1, 0, 0), detection(1, 1, 0)]     # 60s apart
        assert len(TrajectoryEngine(revisit_gap_seconds=120).build("X", near).points) == 1
        assert len(TrajectoryEngine(revisit_gap_seconds=30).build("X", near).points) == 2


class TestMeasurements:
    def test_distance_and_bearing_between_real_sites(self):
        # India Gate -> Connaught Place is roughly 2.4 km, heading NW.
        distance = haversine_km(28.6129, 77.2295, 28.6315, 77.2167)
        assert distance == pytest.approx(2.42, abs=0.1)
        bearing = initial_bearing_degrees(28.6129, 77.2295, 28.6315, 77.2167)
        assert 300 < bearing < 340

    def test_legs_measure_each_hop(self):
        trajectory = TrajectoryEngine().build(
            "X", [detection(o, m) for o, m in zip([1, 2, 3], [0, 12, 25])]
        )
        assert len(trajectory.legs) == 2
        assert all(leg.distance_km > 0 for leg in trajectory.legs)
        assert all(leg.speed_kmh > 0 for leg in trajectory.legs)
        assert trajectory.total_distance_km == pytest.approx(
            sum(leg.distance_km for leg in trajectory.legs), abs=1e-3
        )

    def test_distance_is_none_when_a_camera_is_unsurveyed(self):
        """None, not 0.0 -- "we could not measure" must stay distinguishable
        from "it did not move"."""
        detections = [detection(1, 0), detection(2, 10, latitude=None, longitude=None)]
        leg = TrajectoryEngine().build("X", detections).legs[0]
        assert leg.distance_km is None
        assert leg.speed_kmh is None
        assert leg.duration_seconds == pytest.approx(600)  # clocks still work

    def test_zero_duration_does_not_divide_by_zero(self):
        """Two reads in the same second must not report an infinite speed."""
        detections = [detection(1, 0, 0), detection(2, 0, 0)]
        leg = TrajectoryEngine().build("X", detections).legs[0]
        assert leg.duration_seconds == 0
        assert leg.speed_kmh is None

    def test_total_distance_is_none_when_nothing_is_measurable(self):
        detections = [
            detection(o, m, latitude=None, longitude=None)
            for o, m in zip([1, 2], [0, 10])
        ]
        assert TrajectoryEngine().build("X", detections).total_distance_km is None

    def test_summary_figures(self):
        trajectory = TrajectoryEngine().build(
            "X", [detection(o, m, confidence=c)
                  for o, m, c in zip([1, 2, 3, 4], [0, 12, 25, 48], [0.8, 0.9, 1.0, 0.9])]
        )
        assert trajectory.cameras_visited == 4
        assert trajectory.duration_seconds == pytest.approx(48 * 60)
        assert trajectory.average_confidence == pytest.approx(0.9)
        assert trajectory.first_seen.endswith("09:00:00+00:00")
        assert trajectory.last_seen.endswith("09:48:00+00:00")


class TestEmptyAndDegenerate:
    def test_no_detections_gives_an_empty_trajectory_not_an_error(self):
        trajectory = TrajectoryEngine().build("GHOST", [])
        assert trajectory.is_empty
        assert trajectory.points == []
        assert trajectory.total_distance_km is None
        assert trajectory.duration_seconds is None

    def test_single_detection_has_no_legs(self):
        trajectory = TrajectoryEngine().build("X", [detection(1, 0)])
        assert len(trajectory.points) == 1
        assert trajectory.legs == []
        assert trajectory.duration_seconds is None

    def test_plate_is_normalised(self):
        assert TrajectoryEngine().build(" dl8ca1234 ", [detection(1, 0)]).plate_number == "DL8CA1234"

    def test_missing_camera_identity_does_not_crash(self):
        detections = [detection(1, 0, camera_id=None, camera_name=None)]
        point = TrajectoryEngine().build("X", detections).points[0]
        assert point.camera_id == "UNKNOWN"
        assert point.camera_name  # some renderable label


class TestGeoJSON:
    def test_axis_order_is_longitude_latitude(self):
        """GeoJSON is [lon, lat] -- the reverse of Leaflet. Getting this
        backwards plots Delhi in the Indian Ocean."""
        trajectory = TrajectoryEngine().build("X", [detection(1, 0), detection(2, 10)])
        geojson = trajectory_to_geojson(trajectory)
        line = geojson["features"][0]
        assert line["geometry"]["type"] == "LineString"
        first_lon, first_lat = line["geometry"]["coordinates"][0]
        assert first_lon == pytest.approx(77.2295)   # longitude first
        assert first_lat == pytest.approx(28.6129)

    def test_leaflet_helper_returns_latitude_longitude(self):
        trajectory = TrajectoryEngine().build("X", [detection(1, 0)])
        assert ordered_coordinates(trajectory) == [[pytest.approx(28.6129), pytest.approx(77.2295)]]

    def test_one_point_feature_per_visit_plus_the_line(self):
        trajectory = TrajectoryEngine().build(
            "X", [detection(o, m) for o, m in zip([1, 2, 3, 4], [0, 12, 25, 48])]
        )
        geojson = trajectory_to_geojson(trajectory)
        kinds = [f["properties"]["kind"] for f in geojson["features"]]
        assert kinds == ["path"] + ["detection"] * 4

    def test_a_single_point_produces_no_linestring(self):
        """One sighting is a point, not a path -- and a one-position
        LineString is not valid GeoJSON."""
        geojson = trajectory_to_geojson(TrajectoryEngine().build("X", [detection(1, 0)]))
        assert [f["geometry"]["type"] for f in geojson["features"]] == ["Point"]

    def test_unmapped_points_are_excluded_from_geometry_but_kept_in_the_timeline(self):
        detections = [
            detection(1, 0),
            detection(2, 10, latitude=None, longitude=None),
            detection(3, 20),
        ]
        trajectory = TrajectoryEngine().build("X", detections)
        geojson = trajectory_to_geojson(trajectory)
        assert geojson["properties"]["mapped_points"] == 2
        assert geojson["properties"]["unmapped_points"] == 1
        # The unmapped camera is still on the timeline -- it saw the vehicle.
        assert len(ordered_locations(trajectory)) == 3
        assert [loc["has_location"] for loc in ordered_locations(trajectory)] == [True, False, True]

    def test_empty_trajectory_still_produces_valid_geojson(self):
        """A client must never have to special-case the response shape."""
        geojson = trajectory_to_geojson(TrajectoryEngine().build("GHOST", []))
        assert geojson["type"] == "FeatureCollection"
        assert geojson["features"] == []
        assert bounding_box(TrajectoryEngine().build("GHOST", [])) is None

    def test_popup_properties_carry_everything_a_marker_shows(self):
        trajectory = TrajectoryEngine().build("DL8CA1234", [detection(1, 0)])
        properties = trajectory_to_geojson(trajectory)["features"][0]["properties"]
        for key in (
            "camera_name", "plate_number", "timestamp", "confidence",
            "vehicle_image_path", "detection_count",
        ):
            assert key in properties, f"popup needs {key}"

    def test_bounding_box_spans_the_mapped_points(self):
        trajectory = TrajectoryEngine().build(
            "X", [detection(o, m) for o, m in zip([1, 4], [0, 48])]
        )
        min_lon, min_lat, max_lon, max_lat = bounding_box(trajectory)
        assert min_lat < max_lat and min_lon < max_lon
        assert min_lat == pytest.approx(28.6129)
        assert max_lat == pytest.approx(28.6675)
