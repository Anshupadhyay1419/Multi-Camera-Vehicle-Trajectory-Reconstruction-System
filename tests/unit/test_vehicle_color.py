"""
Unit tests for vehicle body-colour detection (classification/vehicle_color.py).

Synthetic vehicles: a coloured body with a dark windscreen, a black tyre and
a background, plus sensor-like noise -- enough of the distractors a real
tracker crop contains to catch a naive per-pixel vote.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from src.classification.vehicle_color import (
    SUPPORTED_COLORS,
    UNKNOWN,
    VehicleColorConfig,
    VehicleColorDetector,
)

BODY_BGR = {
    "White": (240, 240, 240),
    "Black": (25, 25, 28),
    "Silver": (170, 170, 175),
    "Gray": (105, 105, 108),
    "Blue": (170, 70, 20),
    "Red": (30, 30, 200),
    "Green": (40, 150, 40),
    "Yellow": (20, 210, 230),
    "Brown": (30, 70, 120),
    "Orange": (20, 130, 245),
}


def synthetic_vehicle(body_bgr, height=300, width=400, seed=0):
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = (90, 110, 95)                                             # background
    cv2.rectangle(image, (int(width * .07), int(height * .17)),
                  (int(width * .93), int(height * .90)), body_bgr, -1)   # body
    cv2.rectangle(image, (int(width * .27), int(height * .23)),
                  (int(width * .72), int(height * .43)), (40, 40, 45), -1)  # windscreen
    cv2.rectangle(image, (int(width * .10), int(height * .83)),
                  (int(width * .25), height), (20, 20, 20), -1)          # tyre
    noise = np.random.default_rng(seed).normal(0, 6, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def detector():
    return VehicleColorDetector()


class TestSupportedColours:
    def test_exactly_the_ten_required_colours(self):
        assert set(SUPPORTED_COLORS) == set(BODY_BGR)

    @pytest.mark.parametrize("expected", list(BODY_BGR))
    def test_each_colour_is_recognised(self, detector, expected):
        assert detector.detect(synthetic_vehicle(BODY_BGR[expected])) == expected

    @pytest.mark.parametrize("size", [(120, 160), (300, 400), (720, 1280)])
    def test_result_does_not_depend_on_crop_size(self, detector, size):
        for name, bgr in BODY_BGR.items():
            assert detector.detect(synthetic_vehicle(bgr, *size)) == name, (name, size)

    def test_a_maroon_car_is_red_not_brown(self, detector):
        """Brown is dark ORANGE; dark red stays red."""
        assert detector.detect(synthetic_vehicle((35, 25, 120))) == "Red"

    def test_the_windscreen_and_roof_are_not_read_as_paint(self, detector):
        """Glass sits in the top of a vehicle box and reflects the sky or
        shows a dark cabin; the region skips it, so a white car with a large
        dark windscreen is still white."""
        image = synthetic_vehicle((235, 235, 235))
        cv2.rectangle(image, (40, 50), (360, 100), (30, 30, 30), -1)    # windscreen band
        assert detector.detect(image) == "White"

    def test_chrome_and_glare_do_not_turn_a_black_car_silver(self, detector):
        """Real-footage failure mode: highlights on a black car used to push
        an upper brightness percentile into the silver band."""
        image = synthetic_vehicle((22, 22, 24))
        cv2.rectangle(image, (120, 150), (280, 185), (235, 235, 235), -1)   # chrome grille
        cv2.circle(image, (80, 160), 18, (255, 255, 255), -1)              # headlight
        cv2.circle(image, (320, 160), 18, (255, 255, 255), -1)
        assert detector.detect(image) == "Black"

    def test_dark_navy_paint_is_blue_not_grey(self, detector):
        """Real-footage failure mode: dark saturated paint fell below the
        saturation bar and was classified on brightness instead."""
        assert detector.detect(synthetic_vehicle((95, 45, 25))) == "Blue"


class TestRobustness:
    @pytest.mark.parametrize("crop", [None, np.zeros((0, 0, 3), np.uint8),
                                      np.zeros((5, 5, 3), np.uint8)])
    def test_unusable_input_is_unknown_not_an_error(self, detector, crop):
        assert detector.detect(crop) == UNKNOWN

    def test_a_grayscale_crop_is_handled(self, detector):
        assert detector.detect(np.full((100, 100), 240, np.uint8)) == "White"

    def test_detect_never_raises(self, detector):
        assert detector.detect("not an image") == UNKNOWN

    def test_shares_are_reported(self, detector):
        colour, shares = detector.detect_with_shares(synthetic_vehicle(BODY_BGR["Blue"]))
        assert colour == "Blue"
        assert shares["Blue"] >= 0.9


class TestConfiguration:
    def test_thresholds_come_from_config(self):
        gray_car = synthetic_vehicle(BODY_BGR["Gray"])
        assert VehicleColorDetector().detect(gray_car) == "Gray"
        # A site that calls this brightness silver can say so in config.yaml.
        lenient = VehicleColorDetector.from_config({"vehicle_color": {"silver_min_value": 90}})
        assert lenient.detect(gray_car) == "Silver"

    def test_unknown_config_keys_are_ignored(self):
        config = VehicleColorConfig.from_config({"vehicle_color": {"nonsense": 1}})
        assert config.min_saturation == VehicleColorConfig().min_saturation

    def test_missing_config_uses_defaults(self):
        assert VehicleColorDetector.from_config({}).config == VehicleColorConfig()


class TestPerformance:
    def test_a_full_hd_crop_costs_a_few_milliseconds_at_most(self, detector):
        """Runs once per stored vehicle on the pipeline thread."""
        image = synthetic_vehicle(BODY_BGR["Red"], 1080, 1920)
        detector.detect(image)
        started = time.perf_counter()
        for _ in range(50):
            detector.detect(image)
        per_call_ms = (time.perf_counter() - started) / 50 * 1000
        assert per_call_ms < 5.0, f"{per_call_ms:.2f} ms per vehicle"


class TestTallVehicles:
    """A van's glass sits inside the sampled band, a car's does not.

    region_top skips the top third of the box because that is where the
    glass is -- on a car. A van is taller and boxier, so its rear window
    lands squarely in the middle of what is sampled, and being full of dark
    reflections it outvotes the paint around it. A real white Omni measured
    179 against the 180 needed for White and was stored as Silver, so a
    search for white vehicles did not return it.
    """

    @staticmethod
    def _van(body_bgr, seed=1):
        """A tall vehicle whose rear window falls in the middle of the crop.

        The reflections in the glass are drawn as bands rather than
        per-pixel noise on purpose: the classifier downscales before it
        measures, and noise simply averages back out into a smooth mid-grey
        that reads as bodywork. Real glass carries trees, sky and road --
        structure coarse enough to survive the resize -- and that is what
        makes a glass row distinguishable from a painted one.
        """
        rng = np.random.default_rng(seed)
        image = np.zeros((300, 300, 3), np.uint8)
        image[:] = (90, 110, 95)
        cv2.rectangle(image, (30, 10), (270, 290), body_bgr, -1)
        cv2.rectangle(image, (50, 60), (250, 205), (35, 35, 35), -1)
        for x in range(55, 250, 34):
            cv2.rectangle(image, (x, 62), (x + 17, 203), (150, 155, 150), -1)
        noise = rng.normal(0, 3, image.shape)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def test_a_white_van_is_white_not_silver(self, detector):
        assert detector.detect(self._van(BODY_BGR["White"])) == "White"

    def test_a_black_van_is_still_black(self, detector):
        """The repair must not simply brighten everything."""
        assert detector.detect(self._van(BODY_BGR["Black"])) == "Black"

    def test_a_silver_van_is_still_silver(self, detector):
        assert detector.detect(self._van(BODY_BGR["Silver"])) == "Silver"

    def test_a_gray_van_is_still_gray(self, detector):
        assert detector.detect(self._van(BODY_BGR["Gray"])) == "Gray"

    def test_without_the_repair_the_white_van_reads_silver(self):
        """Pins the bug itself, so the fix cannot be quietly undone.

        panel_row_spread_max=0 counts every row, glass included, which is
        what the classifier used to do -- and it gets the van wrong.
        """
        blind = VehicleColorDetector.from_config(
            {"vehicle_color": {"panel_row_spread_max": 0}}
        )
        assert blind.detect(self._van(BODY_BGR["White"])) == "Silver"
