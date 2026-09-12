"""
Trajectory reconstruction for the multi-camera ALPR system.

Reconstructs where a vehicle went from the detections stored against its
plate, using the coordinates recorded on each event -- no route is
hardcoded anywhere.

    from src.trajectory import TrajectoryEngine, trajectory_to_geojson

    engine = TrajectoryEngine.from_registry(registry)
    with database.get_session() as session:
        detections = database.get_plate_detections(session, "DL8CA1234")
    trajectory = engine.build("DL8CA1234", detections)
    geojson = trajectory_to_geojson(trajectory)
"""

from src.trajectory.engine import (
    ORDER_AUTO,
    ORDER_TIMESTAMP,
    ORDER_TRAJECTORY,
    TrajectoryEngine,
    haversine_km,
    initial_bearing_degrees,
)
from src.trajectory.geojson import (
    bounding_box,
    ordered_coordinates,
    ordered_locations,
    trajectory_to_geojson,
)
from src.trajectory.models import Trajectory, TrajectoryLeg, TrajectoryPoint

__all__ = [
    "TrajectoryEngine",
    "Trajectory",
    "TrajectoryPoint",
    "TrajectoryLeg",
    "trajectory_to_geojson",
    "ordered_coordinates",
    "ordered_locations",
    "bounding_box",
    "haversine_km",
    "initial_bearing_degrees",
    "ORDER_AUTO",
    "ORDER_TIMESTAMP",
    "ORDER_TRAJECTORY",
]
