"""Blacklisted vehicles: storage, matching, detection query, live alert, dashboard."""

from __future__ import annotations

import pytest

from src.alerts.blacklist import Blacklist, normalize_plate
from src.database import db as database


@pytest.fixture()
def blacklist(tmp_path):
    path = tmp_path / "blacklist.yaml"
    path.write_text('plates:\n  - plate: "DL7CD5017"\n    reason: "Flagged"\n')
    return Blacklist(path)


class TestBlacklist:
    def test_the_shipped_file_loads(self):
        """config/blacklist.yaml is edited by the operator from the
        dashboard, so its CONTENTS are runtime state, not something a test
        can pin. What must always hold is that the shipped file parses and
        answers questions -- a malformed one would break every detection."""
        assert Blacklist().contains("DL7CD5017") in (True, False)

    def test_matching_ignores_case_and_spaces(self, blacklist):
        assert blacklist.contains("dl 7cd-5017")
        assert not blacklist.contains("DL7CD5018")
        assert normalize_plate(" dl7 cd5017 ") == "DL7CD5017"

    def test_add_and_remove_persist(self, blacklist):
        blacklist.add("hr26dk8337", "Stolen")
        assert Blacklist(blacklist.path).get("HR26DK8337")["reason"] == "Stolen"
        assert blacklist.remove("HR26DK8337")
        assert not Blacklist(blacklist.path).contains("HR26DK8337")

    def test_invalid_plate_is_rejected(self, blacklist):
        with pytest.raises(ValueError):
            blacklist.add("!!", "x")

    def test_edits_on_disk_are_picked_up(self, blacklist):
        assert blacklist.contains("DL7CD5017")
        import os, time
        time.sleep(0.01)
        blacklist.path.write_text('plates:\n  - "MH01AB0001"\n')
        os.utime(blacklist.path, None)
        blacklist._mtime = None
        assert blacklist.contains("MH01AB0001") and not blacklist.contains("DL7CD5017")


class TestDetections:
    def test_sightings_of_blacklisted_plates_newest_first(self, tmp_path):
        database.init_db(str(tmp_path / "bl.db"))
        try:
            with database.get_session() as session:
                for plate, minute in (("DL7CD5017", 1), ("KA02MH7256", 2), ("DL7CD5017", 3)):
                    database.insert_event(session, {
                        "plate_number": plate, "vehicle_type": "Private", "plate_color": "White",
                        "series_type": "normal", "direction": "IN", "image_path": "",
                        "camera_id": "CAM001", "camera_name": "India Gate",
                        "timestamp": f"2026-09-13T10:0{minute}:00+00:00"})
            with database.get_session() as session:
                rows = database.get_blacklisted_detections(session, {"DL7CD5017"})
            assert [r["timestamp"][14:16] for r in rows] == ["03", "01"]
        finally:
            database._engine = database._SessionFactory = database._db_type = None


class TestLiveAlert:
    def test_detecting_a_blacklisted_plate_logs_an_alert(self, monkeypatch, tmp_path):
        import threading

        from src.alerts import blacklist as module
        from src.cameras.manager import ProgressReporter
        from src.cameras.models import CameraProgress

        path = tmp_path / "b.yaml"
        path.write_text('plates:\n  - plate: "DL7CD5017"\n    reason: "Flagged"\n')
        monkeypatch.setattr(module, "_default", Blacklist(path))

        logs = []
        reporter = ProgressReporter(CameraProgress("CAM004", "Kashmere Gate", 4),
                                    publish=lambda: None, stop_event=threading.Event(),
                                    log=logs.append)
        reporter.on_detection(plate_number="KA02MH7256")
        reporter.on_detection(plate_number="DL7CD5017")
        alerts = [line for line in logs if "BLACKLISTED" in line]
        assert alerts == ["ALERT: BLACKLISTED vehicle DL7CD5017 (Flagged) detected at Kashmere Gate"]
