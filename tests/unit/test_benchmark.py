import pytest

from src.utils.benchmark import BenchmarkRecorder


def test_benchmark_summary_reports_expected_values():
    recorder = BenchmarkRecorder()
    recorder.record_frame(0.1)
    recorder.record_ocr(0.02)
    recorder.record_plate_detection(0.03)

    summary = recorder.summary()

    assert summary["processed_frames"] == 1
    assert summary["fps"] == pytest.approx(10.0, abs=1e-6)
    assert summary["vehicle_detection_latency_ms"]["avg_ms"] == pytest.approx(0.0)
    assert summary["ocr_latency_ms"]["avg_ms"] == pytest.approx(20.0)
    assert summary["plate_detection_latency_ms"]["avg_ms"] == pytest.approx(30.0)
