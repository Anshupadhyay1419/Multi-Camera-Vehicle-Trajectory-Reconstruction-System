"""
Map rendering for reconstructed trajectories.

    from src.mapping import render_trajectory_map
    html = render_trajectory_map(trajectory)
"""

from src.mapping.leaflet import render_trajectory_map

__all__ = ["render_trajectory_map"]
