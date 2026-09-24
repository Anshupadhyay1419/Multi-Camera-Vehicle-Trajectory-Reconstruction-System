"""Validation rules on the registry schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentinel_system.registry.enums import CameraType, Protocol
from sentinel_system.registry.schemas import (
    CameraFilter,
    CameraUpdate,
)
from tests.sentinel_registry.conftest import camera_payload


class TestCameraCode:
    def test_a_code_is_upper_cased_and_trimmed(self):
        """Operators type these off a label; case is not identity."""
        assert camera_payload(camera_code="  ahm-sat-0142 ").camera_code == "AHM-SAT-0142"

    @pytest.mark.parametrize(
        "bad",
        [
            "AB",              # too short
            "-AHM-1",          # must start alphanumeric
            "AHM-1-",          # must end alphanumeric
            "AHM 001",         # no spaces
            "AHM/001",         # no punctuation beyond - and _
            "",
        ],
    )
    def test_malformed_codes_are_refused(self, bad):
        with pytest.raises(ValidationError):
            camera_payload(camera_code=bad)


class TestGeography:
    @pytest.mark.parametrize("lat", [-90.0, 0.0, 23.0225, 90.0])
    def test_latitudes_in_range_are_accepted(self, lat):
        assert camera_payload(latitude=lat).latitude == lat

    @pytest.mark.parametrize("lat", [-90.001, 90.001, 1000.0])
    def test_latitudes_out_of_range_are_refused(self, lat):
        with pytest.raises(ValidationError):
            camera_payload(latitude=lat)

    @pytest.mark.parametrize("lon", [-180.001, 180.001])
    def test_longitudes_out_of_range_are_refused(self, lon):
        with pytest.raises(ValidationError):
            camera_payload(longitude=lon)

    def test_bearing_360_is_refused_because_it_is_zero(self):
        """0 and 360 are the same heading; allowing both splits the data."""
        with pytest.raises(ValidationError):
            camera_payload(bearing_deg=360.0)
        assert camera_payload(bearing_deg=0.0).bearing_deg == 0.0
        assert camera_payload(bearing_deg=359.9).bearing_deg == 359.9

    def test_negative_coverage_radius_is_refused(self):
        with pytest.raises(ValidationError):
            camera_payload(coverage_radius_m=-1.0)


class TestStreamUrl:
    def test_scheme_must_match_the_declared_protocol(self):
        """A stream reader picks its client off `protocol`; a contradiction
        here would only fail later, at connect time."""
        with pytest.raises(ValidationError, match="does not match protocol"):
            camera_payload(protocol=Protocol.RTSP, stream_url="https://cam/stream")

    def test_onvif_accepts_an_rtsp_media_url(self):
        """ONVIF is device management; the media it hands back is RTSP."""
        camera = camera_payload(
            protocol=Protocol.ONVIF, stream_url="rtsp://10.20.4.11:554/onvif1"
        )
        assert camera.stream_url.startswith("rtsp://")

    def test_a_url_without_a_scheme_is_refused(self):
        with pytest.raises(ValidationError, match="scheme"):
            camera_payload(stream_url="10.20.4.11:554/Streaming")

    def test_a_url_without_a_host_is_refused(self):
        with pytest.raises(ValidationError, match="host"):
            camera_payload(stream_url="rtsp:///Streaming")


class TestResolution:
    @pytest.mark.parametrize(
        "raw,expected",
        [("1920x1080", "1920x1080"), ("1920X1080", "1920x1080"), ("1920*1080", "1920x1080")],
    )
    def test_common_spellings_normalise(self, raw, expected):
        assert camera_payload(resolution=raw).resolution == expected

    @pytest.mark.parametrize("bad", ["1080p", "1920", "very big", "x"])
    def test_malformed_resolutions_are_refused(self, bad):
        with pytest.raises(ValidationError):
            camera_payload(resolution=bad)


class TestCrossFieldRules:
    def test_a_ptz_camera_must_declare_ptz_support(self):
        with pytest.raises(ValidationError, match="supports_ptz"):
            camera_payload(camera_type=CameraType.PTZ, supports_ptz=False)

    def test_a_ptz_camera_that_declares_support_is_fine(self):
        camera = camera_payload(camera_type=CameraType.PTZ, supports_ptz=True)
        assert camera.supports_ptz is True


class TestUnknownFields:
    def test_an_unknown_field_is_refused_rather_than_ignored(self):
        """A typo'd field name that is silently dropped is a data-loss bug."""
        with pytest.raises(ValidationError):
            camera_payload(lattitude=23.0)


class TestCameraUpdate:
    def test_omitted_fields_are_not_reported_as_changed(self):
        update = CameraUpdate(camera_name="New name")
        assert update.changed_fields() == {"camera_name": "New name"}

    def test_an_explicit_null_is_a_change_but_an_omission_is_not(self):
        """The distinction a partial update exists to make."""
        assert CameraUpdate(owner=None).changed_fields() == {"owner": None}
        assert CameraUpdate().changed_fields() == {}

    def test_camera_code_cannot_be_patched(self):
        """It is printed on the hardware and quoted in incident reports."""
        with pytest.raises(ValidationError):
            CameraUpdate(camera_code="AHM-NEW-0001")


class TestCameraFilter:
    def test_order_by_is_restricted_to_an_allow_list(self):
        """order_by reaches a SQL ORDER BY clause."""
        with pytest.raises(ValidationError, match="order_by must be one of"):
            CameraFilter(order_by="camera_code; DROP TABLE cameras")

    def test_orderable_is_a_constant_not_a_filter_field(self):
        assert "ORDERABLE" not in CameraFilter.model_fields

    def test_limit_is_capped(self):
        with pytest.raises(ValidationError):
            CameraFilter(limit=10_000)

    def test_defaults_are_a_usable_first_page(self):
        filters = CameraFilter()
        assert (filters.limit, filters.offset, filters.order_by) == (50, 0, "camera_code")
