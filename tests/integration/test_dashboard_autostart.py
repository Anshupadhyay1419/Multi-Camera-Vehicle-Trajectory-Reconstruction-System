"""Autostart: opening the dashboard starts processing -- once, and only when
the dashboard process asks for it. A fake runner stands in for the pipeline,
so none of this can start a GPU session."""

from __future__ import annotations

import dataclasses
import time

import pytest

from src.cameras.manager import CameraManager
from src.cameras.registry import load_camera_registry
from src.cameras.status_store import StatusStore


class UntilStop:
    def __init__(self):
        self.calls = 0

    def __call__(self, config, camera_meta, session_context, progress,
                 source_override, max_duration_seconds=None, models=None):
        self.calls += 1
        progress.on_start(total_frames=0)
        while not progress.should_stop():
            time.sleep(0.01)
        return {"frames_processed": 1, "events_stored": 0}


@pytest.fixture()
def page(monkeypatch, tmp_path, camera_config_file):
    """The dashboard module with its manager swapped for a safe one."""
    import src.dashboard.trajectory_app as app

    registry = load_camera_registry(str(camera_config_file))
    video = tmp_path / "clip.mp4"
    import cv2, numpy as np
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for i in range(5):
        writer.write(np.full((24, 32, 3), i * 40, np.uint8))
    writer.release()
    for camera in registry.all:
        registry.replace(dataclasses.replace(camera, video_path=str(video)))
    registry._settings["processing"] = {"loop_recorded": True,
                                        "upload_dir": str(tmp_path / "none")}
    runner = UntilStop()
    manager = CameraManager({"video": {}}, registry, runner,
                            StatusStore(str(tmp_path / "status.json")))
    from src.cameras.manager import reset_autostart

    monkeypatch.setattr(app, "_manager", lambda: manager)
    reset_autostart()
    yield app, manager, runner
    manager.stop()
    manager.wait(timeout=5)
    reset_autostart()


def test_off_unless_the_dashboard_process_asks(page, monkeypatch):
    """The camera config is shared with the API, CLI and tests; reading it
    must never start a session."""
    app, manager, _ = page
    monkeypatch.delenv("ALPR_AUTOSTART", raising=False)
    assert app._autostart_once() == "disabled"
    assert not manager.is_running()


def test_opening_the_dashboard_starts_processing(page, monkeypatch):
    app, manager, runner = page
    monkeypatch.setenv("ALPR_AUTOSTART", "1")
    assert app._autostart_once() == "started"
    assert manager.is_running()


def test_it_happens_once_so_stop_stays_stopped(page, monkeypatch):
    """Streamlit reruns the script on every click; a restart after STOP
    would make STOP useless."""
    app, manager, runner = page
    monkeypatch.setenv("ALPR_AUTOSTART", "1")
    app._autostart_once()
    manager.stop()
    assert manager.wait(timeout=5)
    app._autostart_once()                  # the next page rerun
    assert not manager.is_running()


def test_nothing_to_run_is_not_an_error(page, monkeypatch):
    app, manager, _ = page
    monkeypatch.setenv("ALPR_AUTOSTART", "1")
    for camera in manager.registry.all:
        manager.registry.replace(dataclasses.replace(camera, video_path=None))
    assert app._autostart_once() == "no camera has a source"
    assert not manager.is_running()


def test_a_config_reload_does_not_re_arm_it(page, monkeypatch):
    """Replacing the manager (Reload camera config) must not restart a
    session the operator stopped."""
    app, manager, runner = page
    monkeypatch.setenv("ALPR_AUTOSTART", "1")
    app._autostart_once()
    manager.stop()
    manager.wait(timeout=5)
    import streamlit as st
    st.cache_resource.clear()          # what a reload / cache clear does
    app._autostart_once()
    assert not manager.is_running()
