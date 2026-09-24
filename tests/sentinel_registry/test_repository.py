"""Repository: queries, filtering, pagination, ordering."""

from __future__ import annotations

import uuid

from sentinel_system.registry.enums import CameraStatus, HealthStatus, Protocol
from sentinel_system.registry.models import Camera
from sentinel_system.registry.schemas import CameraFilter
from tests.sentinel_registry.conftest import camera_payload


def _add(repository, **overrides) -> Camera:
    camera = Camera(**camera_payload(**overrides).model_dump())
    return repository.add(camera)


class TestWrites:
    def test_add_assigns_a_primary_key_without_committing(self, repository, session):
        """Flush, not commit: the caller owns the transaction boundary."""
        camera = _add(repository)
        assert camera.id is not None
        assert session.in_transaction()

    def test_update_applies_only_the_given_fields(self, repository):
        camera = _add(repository)
        original_name = camera.camera_name
        repository.update(camera, {"firmware_version": "V6.0.0"})
        assert camera.firmware_version == "V6.0.0"
        assert camera.camera_name == original_name

    def test_delete_removes_the_row(self, repository):
        camera = _add(repository)
        camera_id = camera.id
        repository.delete(camera)
        assert repository.get_by_id(camera_id) is None


class TestLookup:
    def test_get_by_id_returns_none_when_absent(self, repository):
        """None, not an exception -- naming that absence is the service's job."""
        assert repository.get_by_id(uuid.uuid4()) is None

    def test_get_by_code_is_case_insensitive(self, repository):
        """A lookup that misses an existing camera is how duplicates start."""
        _add(repository, camera_code="AHM-SAT-0142")
        assert repository.get_by_code("ahm-sat-0142") is not None
        assert repository.get_by_code("  AHM-sat-0142  ") is not None

    def test_exists_by_code_can_ignore_one_camera(self, repository):
        """What lets an update re-save a camera without colliding with itself."""
        camera = _add(repository, camera_code="AHM-SAT-0142")
        assert repository.exists_by_code("AHM-SAT-0142") is True
        assert repository.exists_by_code("AHM-SAT-0142", exclude_id=camera.id) is False


class TestFiltering:
    def _populate(self, repository):
        _add(repository, camera_code="AHM-001", district="Ahmedabad",
             status=CameraStatus.ACTIVE, health=HealthStatus.HEALTHY)
        _add(repository, camera_code="AHM-002", district="Ahmedabad",
             status=CameraStatus.INACTIVE, health=HealthStatus.UNREACHABLE)
        _add(repository, camera_code="SUR-001", district="Surat",
             status=CameraStatus.ACTIVE, health=HealthStatus.DEGRADED)

    def test_filtering_by_district(self, repository):
        self._populate(repository)
        found = repository.list(CameraFilter(district="Ahmedabad"))
        assert {c.camera_code for c in found} == {"AHM-001", "AHM-002"}

    def test_filters_combine_as_and(self, repository):
        self._populate(repository)
        found = repository.list(
            CameraFilter(district="Ahmedabad", status=CameraStatus.ACTIVE)
        )
        assert [c.camera_code for c in found] == ["AHM-001"]

    def test_the_active_but_unreachable_query_operations_actually_runs(self, repository):
        """Expected to work, isn't -- the combination a dashboard surfaces."""
        self._populate(repository)
        _add(repository, camera_code="SUR-002", district="Surat",
             status=CameraStatus.ACTIVE, health=HealthStatus.UNREACHABLE)
        found = repository.list(
            CameraFilter(status=CameraStatus.ACTIVE, health=HealthStatus.UNREACHABLE)
        )
        assert [c.camera_code for c in found] == ["SUR-002"]

    def test_search_matches_code_or_name_case_insensitively(self, repository):
        _add(repository, camera_code="AHM-001", camera_name="Satellite Circle")
        _add(repository, camera_code="SUR-001", camera_name="Ring Road")
        assert len(repository.list(CameraFilter(search="satellite"))) == 1
        assert len(repository.list(CameraFilter(search="ahm"))) == 1

    def test_search_treats_wildcards_literally(self, repository):
        """An operator pasting a code with % must not trigger a full scan."""
        _add(repository, camera_code="AHM-001", camera_name="Satellite")
        assert repository.list(CameraFilter(search="%")) == []

    def test_count_ignores_pagination_but_honours_filters(self, repository):
        """A page of 1 reporting a total of 3 is the whole point."""
        self._populate(repository)
        filters = CameraFilter(district="Ahmedabad", limit=1)
        assert len(repository.list(filters)) == 1
        assert repository.count(filters) == 2


class TestOrderingAndPaging:
    def _populate(self, repository):
        for code in ["AHM-003", "AHM-001", "AHM-002"]:
            _add(repository, camera_code=code)

    def test_default_order_is_by_code_ascending(self, repository):
        self._populate(repository)
        found = repository.list(CameraFilter())
        assert [c.camera_code for c in found] == ["AHM-001", "AHM-002", "AHM-003"]

    def test_descending_reverses_the_order(self, repository):
        self._populate(repository)
        found = repository.list(CameraFilter(descending=True))
        assert [c.camera_code for c in found] == ["AHM-003", "AHM-002", "AHM-001"]

    def test_pages_do_not_overlap_or_skip(self, repository):
        self._populate(repository)
        first = repository.list(CameraFilter(limit=2, offset=0))
        second = repository.list(CameraFilter(limit=2, offset=2))
        codes = [c.camera_code for c in first] + [c.camera_code for c in second]
        assert codes == ["AHM-001", "AHM-002", "AHM-003"]

    def test_rows_tying_on_the_sort_column_still_page_deterministically(
        self, repository
    ):
        """Without a tiebreak the planner is free to reorder ties between
        queries, so a row can appear on two pages or on none."""
        for code in ["A-001", "A-002", "A-003", "A-004"]:
            _add(repository, camera_code=code, district="Same")
        filters = {"order_by": "district", "limit": 2}
        first = repository.list(CameraFilter(offset=0, **filters))
        second = repository.list(CameraFilter(offset=2, **filters))
        seen = [c.camera_code for c in first] + [c.camera_code for c in second]
        assert len(set(seen)) == 4
