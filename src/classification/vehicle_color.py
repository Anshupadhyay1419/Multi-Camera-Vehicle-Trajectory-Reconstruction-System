"""
Dominant body-colour detection for a cropped vehicle image.

Distinct from classification/color_classifier.py, which reads the colour of
the NUMBER PLATE (white/yellow/green...) to infer a registration category.
This reads the colour of the VEHICLE BODY -- what a person would say when
describing the car -- and reports one of:

    White, Black, Silver, Gray, Blue, Red, Green, Yellow, Brown, Orange

Approach, chosen for speed (it runs once per stored vehicle, on the pipeline
thread, so it has to stay in the low single milliseconds):

1. Keep the central part of the crop. A tracker box includes road at the
   bottom, sky or background at the top and edges, and wheels and tyres at
   the corners -- none of which are the body.
2. Downscale to a few thousand pixels and convert to HSV.
3. Decide whether the vehicle is CHROMATIC (red, blue...) or ACHROMATIC
   (white, silver, grey, black) by the share of saturated pixels. Most cars
   are achromatic, and their windows, shadows and reflections are too, so a
   plain per-pixel vote would call nearly every coloured car grey.
4. Chromatic: vote over hue among the saturated pixels, with brown split off
   as dark orange/red. Achromatic: classify by the typical brightness of the
   unsaturated pixels.

It is a heuristic, not a trained model: lighting, reflections and camera
white balance all move it. That is the right trade-off for a descriptive
field computed inline with no extra model to load, and the thresholds are
configurable (config.yaml `vehicle_color:`) for a site whose lighting
differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

_logger = get_logger("classification.vehicle_color")

SUPPORTED_COLORS = (
    "White", "Black", "Silver", "Gray", "Blue",
    "Red", "Green", "Yellow", "Brown", "Orange",
)
UNKNOWN = "Unknown"
_CODE = {name: index for index, name in enumerate(SUPPORTED_COLORS)}


@dataclass
class VehicleColorConfig:
    """Tunable thresholds. OpenCV HSV: H 0-179, S and V 0-255."""

    # Region of the crop treated as body, as fractions of height / width.
    # The top 35% is skipped: that is where the windscreen, rear window and
    # roof sit, and they reflect sky and surroundings rather than paint.
    # Measured on 36 hand-labelled crops from this site's videos, skipping
    # it moved accuracy from 26/36 to 30/36.
    region_top: float = 0.35
    region_bottom: float = 0.80
    region_left: float = 0.12
    region_right: float = 0.88
    # Working size: the region is downscaled so its width is this many px.
    sample_width: int = 64
    # A pixel is "saturated" (carries hue) above this S and V.
    min_saturation: int = 55     # dark paint (navy, maroon) is only mildly saturated
    min_value: int = 50
    # Share of saturated pixels above which the vehicle counts as coloured.
    chromatic_share: float = 0.25
    # Achromatic brightness bands (see brightness_percentile below).
    white_min_value: int = 180
    silver_min_value: int = 130
    gray_min_value: int = 70
    # Which brightness percentile of the unsaturated pixels decides white /
    # silver / gray / black. The median, now that the glass is outside the
    # region: an upper percentile (tried first) was pulled up by chrome,
    # headlights and sun glare, turning black and dark cars silver on real
    # footage.
    brightness_percentile: float = 50.0
    # A row of the crop counts as BODYWORK only if it is one colour all the
    # way across -- its own p75-p25 spread no wider than this. See
    # _panel_values(). 0 disables the test and uses every row.
    panel_row_spread_max: int = 60
    # A row needs at least this many unsaturated pixels to be judged at all.
    panel_min_row_pixels: int = 4
    # Below this many bodywork rows there is not enough to judge, and the
    # whole region is used instead.
    panel_min_rows: int = 4
    # Orange/red hues darker than this are brown.
    brown_max_value: int = 150
    # Hue boundaries (upper bounds, exclusive), in order around the wheel.
    hue_red_low: int = 8
    hue_orange: int = 20
    hue_yellow: int = 34
    hue_green: int = 85
    hue_blue: int = 135
    hue_red_high: int = 160      # hues at or above this wrap back to red
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "VehicleColorConfig":
        section = (config or {}).get("vehicle_color") or {}
        known = {name for name in cls.__dataclass_fields__ if name != "extra"}
        values = {key: section[key] for key in section if key in known}
        return cls(**values)


class VehicleColorDetector:
    """Detect the dominant body colour of a cropped vehicle."""

    def __init__(self, config: Optional[VehicleColorConfig] = None) -> None:
        self.config = config or VehicleColorConfig()

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "VehicleColorDetector":
        return cls(VehicleColorConfig.from_config(config))

    def detect(self, vehicle_crop: Optional[np.ndarray]) -> str:
        """Return one of SUPPORTED_COLORS, or "Unknown" if it cannot tell.

        Never raises: a colour is descriptive metadata and must never cost the
        pipeline a stored vehicle.
        """
        try:
            return self.detect_with_shares(vehicle_crop)[0]
        except Exception as exc:
            _logger.debug("Vehicle colour detection failed: %s", exc)
            return UNKNOWN

    def detect_with_shares(
        self, vehicle_crop: Optional[np.ndarray]
    ) -> tuple[str, dict[str, float]]:
        """Like detect(), also returning each colour's share of the vote."""
        hsv = self._sample(vehicle_crop)
        if hsv is None:
            return UNKNOWN, {}

        c = self.config
        hue = hsv[..., 0].astype(np.int16)
        sat = hsv[..., 1]
        val = hsv[..., 2]
        total = hue.size

        saturated = (sat >= c.min_saturation) & (val >= c.min_value)
        chromatic_share = float(saturated.sum()) / total

        if chromatic_share >= c.chromatic_share:
            codes = self._hue_codes(hue[saturated], val[saturated])
        else:
            codes = self._brightness_codes(val, ~saturated)

        if codes.size == 0:
            return UNKNOWN, {}
        counts = np.bincount(codes, minlength=len(SUPPORTED_COLORS))
        shares = {
            SUPPORTED_COLORS[i]: round(float(n) / codes.size, 3)
            for i, n in enumerate(counts) if n
        }
        return SUPPORTED_COLORS[int(np.argmax(counts))], shares

    # ── internals ────────────────────────────────────────────────────────

    def _sample(self, crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if crop is None or not hasattr(crop, "shape") or crop.size == 0:
            return None
        if crop.ndim == 2:                       # grayscale: no hue to read
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        height, width = crop.shape[:2]
        if height < 12 or width < 12:
            return None

        c = self.config
        top, bottom = int(height * c.region_top), int(height * c.region_bottom)
        left, right = int(width * c.region_left), int(width * c.region_right)
        region = crop[top:max(bottom, top + 1), left:max(right, left + 1)]
        if region.size == 0:
            return None

        # Skip pixels before the area-average resize. Averaging a 1280px-wide
        # region straight down to 64px dominated the cost (measured 2.8 ms per
        # 720p crop); striding to ~2x the target first and averaging only the
        # rest gives the same colour for a fraction of the work.
        stride = max(1, region.shape[1] // (c.sample_width * 2))
        if stride > 1:
            region = region[::stride, ::stride]

        scale = c.sample_width / float(region.shape[1])
        if scale < 1.0:
            region = cv2.resize(
                region,
                (c.sample_width, max(1, int(round(region.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    def _hue_codes(self, hue: np.ndarray, val: np.ndarray) -> np.ndarray:
        """Per-pixel colour index (into SUPPORTED_COLORS) for saturated pixels.

        Integer codes rather than string labels: this runs inline on the
        pipeline thread for every stored vehicle, and integer bincount over a
        few thousand pixels is an order of magnitude cheaper than comparing
        Python strings.
        """
        c = self.config
        codes = np.full(hue.shape, _CODE["Red"], dtype=np.int8)
        codes[(hue >= c.hue_red_low) & (hue < c.hue_orange)] = _CODE["Orange"]
        codes[(hue >= c.hue_orange) & (hue < c.hue_yellow)] = _CODE["Yellow"]
        codes[(hue >= c.hue_yellow) & (hue < c.hue_green)] = _CODE["Green"]
        # 135-160 is purple/magenta, not a supported colour: the lower half
        # reads as blue, the upper half stays red.
        codes[(hue >= c.hue_green) & (hue < (c.hue_blue + c.hue_red_high) // 2)] = _CODE["Blue"]
        # Brown is DARK ORANGE, not a hue of its own. Deliberately not dark
        # red (0-3) or dark magenta-red (160+): a maroon car is red.
        codes[(hue >= 4) & (hue < c.hue_orange) & (val < c.brown_max_value)] = _CODE["Brown"]
        return codes.astype(np.int64)

    def _panel_values(self, val: np.ndarray, unsaturated: np.ndarray) -> np.ndarray:
        """The unsaturated pixels that are BODYWORK, ignoring glass and trim.

        region_top exists to put the windscreen and rear window above the
        sampled band, and on a car it does. On a tall vehicle it does not: a
        van's rear window sits squarely inside 35-80% of the box, and being
        dark it drags the median of the whole region down. A white Omni
        measured 179 against the 180 needed for White and was reported
        Silver -- the paint was never in question, the glass outvoted it.

        Rather than assume where the glass is, ask which rows look like
        painted panel. A panel row is ONE colour the whole way across, so
        its own quartile spread is tiny; the van's paint rows sit at 247
        with a spread of 0-8. Glass is not: it carries reflections of trees,
        road and sky, and those rows spread 60-120. Trim is not either -- the
        row through a black car's chrome grille and headlights is half
        near-black paint and half blown-out highlight.

        So only uniform rows are counted, and what is left is bodywork. This
        is deliberately not "take a brighter percentile", which is what an
        earlier attempt did: highlights on a dark car pushed it into the
        silver band, because that rule cannot tell a bright PIXEL from a
        bright PANEL. A black car's paint rows are uniform and dark, so they
        are exactly the rows kept, and it stays Black.
        """
        c = self.config
        flat = val[unsaturated]
        if c.panel_row_spread_max <= 0 or val.ndim != 2:
            return flat

        panel_rows = np.zeros(val.shape[0], dtype=bool)
        for index in range(val.shape[0]):
            row = val[index][unsaturated[index]]
            if row.size < c.panel_min_row_pixels:
                continue
            quarter, three = np.percentile(row, [25, 75])
            panel_rows[index] = (three - quarter) <= c.panel_row_spread_max

        if int(panel_rows.sum()) < c.panel_min_rows:
            return flat
        panels = val[panel_rows][unsaturated[panel_rows]]
        return panels if panels.size else flat

    def _brightness_codes(self, val: np.ndarray, unsaturated: np.ndarray) -> np.ndarray:
        if val.size == 0:
            return np.array([], dtype=np.int64)
        c = self.config
        values = self._panel_values(val, unsaturated)
        if values.size == 0:
            return np.array([], dtype=np.int64)
        # One label for the whole vehicle from its typical panel brightness,
        # rather than a per-pixel vote over pixels of very different origin
        # (paint, trim, shadow).
        level = float(np.percentile(values, c.brightness_percentile))
        if level >= c.white_min_value:
            label = "White"
        elif level >= c.silver_min_value:
            label = "Silver"
        elif level >= c.gray_min_value:
            label = "Gray"
        else:
            label = "Black"
        return np.array([_CODE[label]], dtype=np.int64)
