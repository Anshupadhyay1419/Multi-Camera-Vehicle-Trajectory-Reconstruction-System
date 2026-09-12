"""
Unit tests for utils/config.get_camera_metadata().

This resolves the `camera:` block of config.yaml into the four fields that
get stamped onto every stored event (camera_id, camera_name, latitude,
longitude). It is deliberately forgiving -- a hand-edited config should never
stop a gate from recording vehicles -- so most of these tests pin down what
happens for malformed input.

Tests cover:
- A fully populated block passes through unchanged
- A missing block / missing fields fall back to visible placeholders
- Coordinates given as strings are coerced to floats
- Unusable or out-of-range coordinates become None rather than raising
- The returned keys match the VehicleEvent column names exactly
"""

from __future__ import annotations

import pytest

from src.utils.config import get_camera_metadata


EXPECTED_KEYS = {"camera_id", "camera_name", "latitude", "longitude"}


def _config(**camera) -> dict:
    return {"camera": camera}


# ---------------------------------------------------------------------------
# Shape of the return value
# ---------------------------------------------------------------------------

class TestReturnShape:
    def test_returns_exactly_the_event_column_names(self):
        """The result is splatted into an event_data dict, so the keys must
        match VehicleEvent's column names -- no more, no less."""
        result = get_camera_metadata(_config(camera_id="G1", camera_name="Gate 1"))
        assert set(result.keys()) == EXPECTED_KEYS

    def test_ids_are_always_non_empty_strings(self):
        result = get_camera_metadata({})
        assert isinstance(result["camera_id"], str) and result["camera_id"]
        assert isinstance(result["camera_name"], str) and result["camera_name"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestPopulatedBlock:
    def test_full_block_passes_through(self):
        result = get_camera_metadata(_config(
            camera_id="GATE-01", camera_name="Main Gate",
            latitude=28.613939, longitude=77.209023,
        ))
        assert result == {
            "camera_id":   "GATE-01",
            "camera_name": "Main Gate",
            "latitude":    28.613939,
            "longitude":   77.209023,
        }

    def test_ids_are_stripped(self):
        result = get_camera_metadata(_config(
            camera_id="  GATE-01  ", camera_name="  Main Gate  ",
        ))
        assert result["camera_id"] == "GATE-01"
        assert result["camera_name"] == "Main Gate"

    def test_negative_coordinates_are_kept(self):
        """Southern/western hemispheres are valid, and 0 is a real value --
        none of these may be discarded as falsy."""
        result = get_camera_metadata(_config(
            camera_id="G", camera_name="G", latitude=-12.04318, longitude=-77.02824,
        ))
        assert result["latitude"] == pytest.approx(-12.04318)
        assert result["longitude"] == pytest.approx(-77.02824)

    def test_zero_coordinates_are_kept_not_treated_as_missing(self):
        result = get_camera_metadata(_config(
            camera_id="G", camera_name="G", latitude=0, longitude=0,
        ))
        assert result["latitude"] == 0.0
        assert result["longitude"] == 0.0


# ---------------------------------------------------------------------------
# Missing / malformed block
# ---------------------------------------------------------------------------

class TestMissingBlock:
    def test_no_camera_key_yields_placeholders(self):
        """An existing deployment's config.yaml predates the camera block and
        must still load."""
        result = get_camera_metadata({"video": {}, "database": {}})
        assert result["camera_id"] == "UNKNOWN"
        assert result["latitude"] is None
        assert result["longitude"] is None

    def test_empty_config_yields_placeholders(self):
        assert get_camera_metadata({})["camera_id"] == "UNKNOWN"

    def test_null_camera_block_yields_placeholders(self):
        """`camera:` with nothing under it parses as None, not {}."""
        assert get_camera_metadata({"camera": None})["camera_id"] == "UNKNOWN"

    def test_scalar_camera_block_yields_placeholders(self):
        """A flat `camera: GATE-01` instead of a mapping must not raise."""
        assert get_camera_metadata({"camera": "GATE-01"})["camera_id"] == "UNKNOWN"

    def test_blank_id_yields_placeholder(self):
        result = get_camera_metadata(_config(camera_id="   ", camera_name=""))
        assert result["camera_id"] == "UNKNOWN"
        assert result["camera_name"] == "Unconfigured camera"


# ---------------------------------------------------------------------------
# Coordinate coercion
# ---------------------------------------------------------------------------

class TestCoordinateCoercion:
    def test_string_coordinates_are_coerced_to_float(self):
        """YAML quoting is easy to do by accident when hand-editing."""
        result = get_camera_metadata(_config(
            camera_id="G", camera_name="G",
            latitude="28.613939", longitude="77.209023",
        ))
        assert result["latitude"] == pytest.approx(28.613939)
        assert result["longitude"] == pytest.approx(77.209023)

    @pytest.mark.parametrize("bad", ["", "n/a", "TBD", None, [], {}])
    def test_unusable_coordinates_become_none(self, bad):
        result = get_camera_metadata(_config(
            camera_id="G", camera_name="G", latitude=bad, longitude=bad,
        ))
        assert result["latitude"] is None
        assert result["longitude"] is None

    def test_out_of_range_latitude_is_rejected(self):
        """A latitude past the pole is a config error (usually lat/lon
        swapped); a wrong coordinate is worse than none."""
        result = get_camera_metadata(_config(
            camera_id="G", camera_name="G", latitude=177.2, longitude=28.6,
        ))
        assert result["latitude"] is None
        assert result["longitude"] == pytest.approx(28.6)

    def test_out_of_range_longitude_is_rejected(self):
        result = get_camera_metadata(_config(
            camera_id="G", camera_name="G", latitude=28.6, longitude=277.2,
        ))
        assert result["longitude"] is None
        assert result["latitude"] == pytest.approx(28.6)

    @pytest.mark.parametrize("lat,lon", [(90.0, 180.0), (-90.0, -180.0)])
    def test_range_boundaries_are_accepted(self, lat, lon):
        result = get_camera_metadata(_config(
            camera_id="G", camera_name="G", latitude=lat, longitude=lon,
        ))
        assert result["latitude"] == lat
        assert result["longitude"] == lon

    def test_bad_latitude_does_not_discard_good_longitude(self):
        result = get_camera_metadata(_config(
            camera_id="G", camera_name="G", latitude="oops", longitude=77.209023,
        ))
        assert result["latitude"] is None
        assert result["longitude"] == pytest.approx(77.209023)


# ---------------------------------------------------------------------------
# The shipped config
# ---------------------------------------------------------------------------

class TestShippedConfig:
    def test_repo_config_has_a_resolvable_camera_block(self):
        from pathlib import Path

        from src.utils.config import load_config

        repo_root = Path(__file__).resolve().parents[2]
        config = load_config(str(repo_root / "config" / "config.yaml"))
        result = get_camera_metadata(config)
        assert set(result.keys()) == EXPECTED_KEYS
        # Shipped config must name a real gate, not fall through to the
        # "block is missing" placeholder.
        assert result["camera_id"] != "UNKNOWN"
