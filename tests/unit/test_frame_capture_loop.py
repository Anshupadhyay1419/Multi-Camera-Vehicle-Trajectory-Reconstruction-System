"""FrameCapture's loop mode: an uploaded clip that behaves like a feed."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.capture.frame_capture import FrameCapture


@pytest.fixture()
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 24))
    if not writer.isOpened():
        pytest.skip("no usable OpenCV video encoder")
    for index in range(10):
        writer.write(np.full((24, 32, 3), index * 20, np.uint8))
    writer.release()
    return str(path)


def _read(capture, limit):
    count = 0
    while count < limit:
        ok, _ = capture.read_frame()
        if not ok:
            break
        count += 1
    return count


def test_a_clip_ends_without_loop(clip):
    capture = FrameCapture(clip, frame_skip=1, realtime_playback=False)
    capture.open()
    try:
        assert _read(capture, 100) == 10
    finally:
        capture.release()


def test_a_looping_clip_never_ends_on_its_own(clip):
    capture = FrameCapture(clip, frame_skip=1, realtime_playback=False, loop=True)
    capture.open()
    try:
        assert _read(capture, 35) == 35
        assert capture.loops_completed == 3
    finally:
        capture.release()


def test_stream_time_never_runs_backwards_across_the_seam(clip):
    """Duplicate suppression works in stream time, so a loop must never make
    the clock jump back to zero.

    Non-decreasing rather than strictly increasing: some backends report
    position 0 for the very first frame of a file, so frames 1 and 2 share a
    timestamp. That predates looping and is harmless; running BACKWARDS is
    the thing that would break the duplicate window.
    """
    capture = FrameCapture(clip, frame_skip=1, realtime_playback=False, loop=True)
    capture.open()
    try:
        times = []
        for _ in range(25):   # 2.5 passes of a 10-frame clip
            capture.read_frame()
            times.append(capture.stream_time_seconds())
        assert times == sorted(times), f"stream time went backwards: {times}"
        clip_seconds = 10 / 30.0
        assert times[-1] > 2 * clip_seconds, "stream time stopped at the seam"
    finally:
        capture.release()


def test_loop_is_ignored_for_live_sources():
    assert FrameCapture("rtsp://10.0.0.5/live", loop=True).loop is False


def test_loop_comes_from_config(clip):
    capture = FrameCapture.from_config({"video": {"source": clip, "loop": True}})
    assert capture.loop is True
