"""Service: use-cases, domain errors and the transaction boundary."""

from __future__ import annotations

import uuid

import pytest

from sentinel_system.registry.enums import CameraStatus, CameraType, HealthStatus
from sentinel_system.registry.exceptions import (
    CameraNotFoundError,
    DuplicateCameraCodeError,
    RegistryError,
)
from sentinel_system.registry.schemas import CameraFilter, CameraRead, CameraUpdate
from tests.sentinel_registry.conftest import camera_payload


class TestCreate:
    def test_create_returns_a_validated_schema_not_an_orm_row(self, service, payload):
        """A detached ORM instance raises on attribute access once its
        session closes; a value object cannot."""
        created = service.create(payload)
        assert isinstance(created, CameraRead)
        assert created.camera_code == "AHM-SAT-0142"
        assert isinstance(created.id, uuid.UUID)

    def test_create_persists_every_field_it_was_given(self, service, payload):
        created = service.create(payload)
        fetched = service.get(created.id)
        assert fetched.camera_name == payload.camera_name
        assert fetched.latitude == payload.latitude
        assert fetched.protocol is payload.protocol
        assert fetched.supports_analytics is True
        assert fetched.coverage_radius_m == 80.0

    def test_create_commits(self, service, session, payload):
        created = service.create(payload)
        session.expire_all()
        assert service.get(created.id).camera_code == created.camera_code

    def test_a_duplicate_code_raises_a_domain_error(self, service, payload):
        service.create(payload)
        with pytest.raises(DuplicateCameraCodeError) as info:
            service.create(camera_payload(camera_name="Another camera"))
        assert info.value.camera_code == "AHM-SAT-0142"
        assert info.value.code == "duplicate_camera_code"

    def test_a_duplicate_differing_only_in_case_is_still_a_duplicate(
        self, service, payload
    ):
        service.create(payload)
        with pytest.raises(DuplicateCameraCodeError):
            service.create(camera_payload(camera_code="ahm-sat-0142"))

    def test_a_failed_create_leaves_the_session_usable(self, service, payload):
        """A rolled-back conflict must not poison the transaction for the
        next request on the same session."""
        service.create(payload)
        with pytest.raises(DuplicateCameraCodeError):
            service.create(camera_payload())
        other = service.create(camera_payload(camera_code="AHM-SAT-0143"))
        assert other.camera_code == "AHM-SAT-0143"


class TestRead:
    def test_get_raises_when_the_id_is_unknown(self, service):
        missing = uuid.uuid4()
        with pytest.raises(CameraNotFoundError) as info:
            service.get(missing)
        assert info.value.camera_id == missing
        assert info.value.code == "camera_not_found"

    def test_get_by_code_is_case_insensitive(self, service, payload):
        service.create(payload)
        assert service.get_by_code("ahm-sat-0142").camera_code == "AHM-SAT-0142"

    def test_get_by_code_raises_when_unknown(self, service):
        with pytest.raises(CameraNotFoundError):
            service.get_by_code("NO-SUCH-0001")

    def test_every_registry_error_shares_one_base(self, service):
        """A caller that does not care which failure it was can still catch
        the platform's errors without swallowing ValueError."""
        with pytest.raises(RegistryError):
            service.get(uuid.uuid4())

    def test_exists_answers_without_raising(self, service, payload):
        assert service.exists("AHM-SAT-0142") is False
        service.create(payload)
        assert service.exists("AHM-SAT-0142") is True


class TestList:
    def test_an_empty_registry_lists_cleanly(self, service):
        page = service.list()
        assert page.items == [] and page.total == 0 and page.has_more is False

    def test_total_counts_matches_not_the_page(self, service):
        for n in range(5):
            service.create(camera_payload(camera_code=f"AHM-{n:03d}"))
        page = service.list(CameraFilter(limit=2))
        assert len(page.items) == 2
        assert page.total == 5
        assert page.has_more is True

    def test_the_last_page_reports_no_more(self, service):
        for n in range(3):
            service.create(camera_payload(camera_code=f"AHM-{n:03d}"))
        page = service.list(CameraFilter(limit=2, offset=2))
        assert len(page.items) == 1 and page.has_more is False

    def test_filters_reach_the_query(self, service):
        service.create(camera_payload(camera_code="AHM-001", district="Ahmedabad"))
        service.create(camera_payload(camera_code="SUR-001", district="Surat"))
        page = service.list(CameraFilter(district="Surat"))
        assert [c.camera_code for c in page.items] == ["SUR-001"]


class TestUpdate:
    def test_a_partial_update_touches_only_what_was_sent(self, service, payload):
        created = service.create(payload)
        updated = service.update(created.id, CameraUpdate(firmware_version="V6.0.0"))
        assert updated.firmware_version == "V6.0.0"
        assert updated.camera_name == created.camera_name
        assert updated.latitude == created.latitude

    def test_an_explicit_null_clears_a_field(self, service, payload):
        created = service.create(payload)
        assert created.owner is not None
        assert service.update(created.id, CameraUpdate(owner=None)).owner is None

    def test_an_empty_update_does_not_touch_updated_at(self, service, payload):
        """An empty PATCH should not make the record look freshly edited."""
        created = service.create(payload)
        unchanged = service.update(created.id, CameraUpdate())
        assert unchanged.updated_at == created.updated_at

    def test_updating_an_unknown_camera_raises(self, service):
        with pytest.raises(CameraNotFoundError):
            service.update(uuid.uuid4(), CameraUpdate(camera_name="X"))

    def test_status_and_health_can_move_independently(self, service, payload):
        """ACTIVE + UNREACHABLE is the state operations most needs to see."""
        created = service.create(payload)
        updated = service.update(
            created.id,
            CameraUpdate(status=CameraStatus.ACTIVE, health=HealthStatus.UNREACHABLE),
        )
        assert updated.status is CameraStatus.ACTIVE
        assert updated.health is HealthStatus.UNREACHABLE


class TestUpdateCrossFieldRules:
    """Rules a payload cannot break alone, only in combination with the row."""

    def test_switching_type_to_ptz_without_ptz_support_is_refused(
        self, service, payload
    ):
        created = service.create(payload)  # supports_ptz=False on the stored row
        with pytest.raises(ValueError, match="supports_ptz"):
            service.update(created.id, CameraUpdate(camera_type=CameraType.PTZ))

    def test_switching_to_ptz_and_enabling_support_together_is_allowed(
        self, service, payload
    ):
        created = service.create(payload)
        updated = service.update(
            created.id,
            CameraUpdate(camera_type=CameraType.PTZ, supports_ptz=True),
        )
        assert updated.camera_type is CameraType.PTZ

    def test_a_new_url_is_checked_against_the_stored_protocol(self, service, payload):
        """The payload alone looks fine; only the merged row is wrong."""
        created = service.create(payload)  # protocol=RTSP on the stored row
        with pytest.raises(ValueError, match="does not match protocol"):
            service.update(created.id, CameraUpdate(stream_url="https://cam/stream"))

    def test_changing_protocol_and_url_together_is_allowed(self, service, payload):
        created = service.create(payload)
        from sentinel_system.registry.enums import Protocol

        updated = service.update(
            created.id,
            CameraUpdate(protocol=Protocol.HTTPS, stream_url="https://cam/stream"),
        )
        assert updated.protocol is Protocol.HTTPS


class TestDelete:
    def test_delete_removes_the_camera(self, service, payload):
        created = service.create(payload)
        service.delete(created.id)
        with pytest.raises(CameraNotFoundError):
            service.get(created.id)

    def test_deleting_an_unknown_camera_raises(self, service):
        with pytest.raises(CameraNotFoundError):
            service.delete(uuid.uuid4())

    def test_the_code_is_free_for_reuse_after_a_delete(self, service, payload):
        created = service.create(payload)
        service.delete(created.id)
        assert service.create(camera_payload()).camera_code == "AHM-SAT-0142"
