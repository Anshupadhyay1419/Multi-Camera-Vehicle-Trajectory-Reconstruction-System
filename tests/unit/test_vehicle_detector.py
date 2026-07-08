"""
Unit tests for src.detection.vehicle_detector.VehicleDetector.
"""

from src.detection.vehicle_detector import Detection, VehicleDetector


class TestVehicleDetectorNonMaxSuppression:
    def test_overlapping_same_class_keeps_highest_confidence_box(self):
        detections = [
            Detection(bbox=(10, 10, 110, 110), class_label="car", confidence=0.91),
            Detection(bbox=(15, 15, 105, 105), class_label="car", confidence=0.84),
            Detection(bbox=(200, 200, 260, 260), class_label="car", confidence=0.77),
        ]

        result = VehicleDetector.non_max_suppression(detections, iou_threshold=0.5)

        assert len(result) == 2
        assert detections[0] in result
        assert detections[2] in result
        assert detections[1] not in result

    def test_overlapping_different_classes_are_processed_independently(self):
        detections = [
            Detection(bbox=(10, 10, 110, 110), class_label="car", confidence=0.91),
            Detection(bbox=(15, 15, 105, 105), class_label="bus", confidence=0.84),
        ]

        result = VehicleDetector.non_max_suppression(detections, iou_threshold=0.5)

        assert len(result) == 2
        assert detections[0] in result
        assert detections[1] in result
