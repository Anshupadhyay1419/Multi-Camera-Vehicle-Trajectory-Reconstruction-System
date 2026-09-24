"""The registry's closed vocabularies."""

from __future__ import annotations

import pytest

from sentinel_system.registry.enums import (
    CameraStatus,
    CameraType,
    Codec,
    HealthStatus,
    MaintenanceStatus,
    Protocol,
)

ALL_ENUMS = [
    CameraStatus,
    HealthStatus,
    MaintenanceStatus,
    CameraType,
    Protocol,
    Codec,
]


class TestValueSemantics:
    @pytest.mark.parametrize("enum_cls", ALL_ENUMS)
    def test_members_are_lowercase_snake_case_strings(self, enum_cls):
        """These values reach URLs, JSON payloads, log lines and SQL. Mixed
        case would make every one of those a place to get it wrong."""
        for member in enum_cls:
            assert member.value == member.value.lower()
            assert " " not in member.value

    @pytest.mark.parametrize("enum_cls", ALL_ENUMS)
    def test_str_renders_the_value_not_the_python_name(self, enum_cls):
        """So an f-string in a log line says 'active', not
        'CameraStatus.ACTIVE'."""
        for member in enum_cls:
            assert str(member) == member.value

    @pytest.mark.parametrize("enum_cls", ALL_ENUMS)
    def test_values_are_unique(self, enum_cls):
        values = [m.value for m in enum_cls]
        assert len(values) == len(set(values))

    @pytest.mark.parametrize("enum_cls", ALL_ENUMS)
    def test_values_helper_lists_every_member(self, enum_cls):
        assert enum_cls.values() == [m.value for m in enum_cls]

    def test_members_compare_equal_to_their_string(self, enum_cls=CameraStatus):
        assert CameraStatus.ACTIVE == "active"


class TestSeparationOfStatusAndHealth:
    def test_status_and_health_are_different_vocabularies(self):
        """An ACTIVE camera can be UNREACHABLE -- expected to work, isn't.
        Collapsing these into one field loses exactly that state."""
        assert set(CameraStatus.values()) & set(HealthStatus.values()) == set()

    def test_health_has_an_unknown_member(self):
        """A camera nobody has checked must not default to healthy."""
        assert HealthStatus.UNKNOWN in HealthStatus

    def test_status_covers_the_whole_lifecycle(self):
        assert {"planned", "active", "decommissioned"} <= set(CameraStatus.values())


class TestProtocolSchemes:
    @pytest.mark.parametrize("protocol", list(Protocol))
    def test_every_protocol_declares_at_least_one_scheme(self, protocol):
        assert len(protocol.url_schemes) >= 1

    def test_rtsp_accepts_its_tls_variant(self):
        assert "rtsps" in Protocol.RTSP.url_schemes

    def test_onvif_accepts_http_and_rtsp(self):
        """ONVIF is device management over HTTP; the media is RTSP."""
        assert {"http", "rtsp"} <= set(Protocol.ONVIF.url_schemes)

    def test_rtsp_does_not_accept_plain_http(self):
        assert "http" not in Protocol.RTSP.url_schemes
