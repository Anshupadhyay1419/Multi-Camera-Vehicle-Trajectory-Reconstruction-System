"""
Map engine: render a reconstructed trajectory as a self-contained Leaflet map.

Produces one HTML document with the trajectory baked in as GeoJSON. Nothing
about the route is hardcoded -- every coordinate comes from the Trajectory,
which in turn comes from the coordinates stored on each event.

Why Leaflet directly rather than Folium: the page has to embed in a
Streamlit component iframe, which is sandboxed and cannot read the server's
filesystem. Thumbnails therefore have to travel inside the document as data
URIs, and the popup markup has to be built here rather than by a Python
templating layer that assumes it can serve files. Doing it directly also
drops two dependencies (folium, streamlit-folium) that are not installed on
the Jetson, and keeps the same renderer usable from FastAPI.

Layers:
  * OpenStreetMap standard tiles (default)
  * Esri World Imagery satellite tiles
  toggled by Leaflet's own layer control.

Drawn:
  * a colour-graded polyline along the path, with a casing underneath so it
    stays readable over both light street tiles and dark satellite imagery
  * direction arrows rotated to each leg's true bearing
  * numbered markers, click for a popup with camera name, plate, timestamp,
    confidence and the vehicle thumbnail
"""

from __future__ import annotations

import base64
import json
import mimetypes
from html import escape
from pathlib import Path
from typing import Optional

from src.trajectory.geojson import trajectory_to_geojson
from src.trajectory.models import Trajectory
from src.utils.logger import get_logger

_logger = get_logger("mapping.leaflet")

# Pinned versions from a CDN with SRI-friendly stable URLs. Pinned rather
# than "latest" so a Leaflet release cannot change the look of a deployed
# dashboard without anyone touching this repository.
_LEAFLET_CSS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css"
_LEAFLET_JS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"

# Fallback view when a trajectory has no mappable points at all: central
# Delhi, wide enough to show the whole camera network. Only ever used for an
# empty map -- any real trajectory is fitted to its own bounds.
_FALLBACK_CENTER = (28.6304, 77.2177)
_FALLBACK_ZOOM = 12

# Largest image to inline as a data URI. Thumbnails are a few KB; anything
# much bigger is a full frame that was mislabelled, and inlining a dozen of
# those would produce a document too large for an iframe to render smoothly.
_MAX_INLINE_IMAGE_BYTES = 512 * 1024


def _encode_image(path: Optional[str], repo_root: Path) -> Optional[str]:
    """Read an image and return it as a data URI, or None if unusable.

    Every failure mode -- missing file, unreadable, too large, not an image
    -- returns None rather than raising: a missing thumbnail must cost the
    popup its picture, never the whole map.
    """
    if not path:
        return None

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate

    try:
        if not candidate.is_file():
            return None
        size = candidate.stat().st_size
        if size == 0 or size > _MAX_INLINE_IMAGE_BYTES:
            _logger.debug("Skipping thumbnail %s (%d bytes)", candidate, size)
            return None
        mime, _ = mimetypes.guess_type(candidate.name)
        if mime is None or not mime.startswith("image/"):
            mime = "image/jpeg"
        encoded = base64.b64encode(candidate.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except OSError as exc:
        _logger.debug("Could not inline thumbnail %s: %s", candidate, exc)
        return None


def _with_thumbnails(geojson: dict, repo_root: Path) -> dict:
    """Replace image *paths* in the GeoJSON with inline data URIs.

    The iframe cannot fetch local paths, so a path that survived to the
    browser would render as a broken image. Converting here keeps the
    GeoJSON producer free of any assumption about how it gets displayed.
    """
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        if properties.get("kind") != "detection":
            continue
        for source_key, target_key in (
            ("vehicle_image_path", "vehicle_image"),
            ("plate_image_path", "plate_image"),
        ):
            properties[target_key] = _encode_image(properties.get(source_key), repo_root)
    return geojson


def render_trajectory_map(
    trajectory: Trajectory,
    repo_root: Optional[Path] = None,
    height: int = 560,
    include_thumbnails: bool = True,
    satellite_default: bool = False,
) -> str:
    """Render *trajectory* as a standalone Leaflet HTML document.

    Args:
        trajectory:         The reconstructed path to draw.
        repo_root:          Base for resolving relative image paths. Defaults
                            to the repository root.
        height:             Map height in CSS pixels.
        include_thumbnails: Inline vehicle/plate crops into the popups. Turn
                            off for a long trajectory where document size
                            matters more than the pictures.
        satellite_default:  Open on Esri imagery instead of OSM streets.

    Returns:
        A complete HTML document. Safe to write to a file, embed with
        Streamlit's `components.html`, or serve from FastAPI.
    """
    repo_root = repo_root or Path(__file__).resolve().parents[2]

    geojson = trajectory_to_geojson(trajectory)
    if include_thumbnails:
        geojson = _with_thumbnails(geojson, repo_root)

    # json.dumps into a <script> is the safe way to pass structured data to
    # the page. The "</" escape stops a string containing "</script>" from
    # closing the tag early -- the one XSS hole this pattern has, and stored
    # OCR text is exactly the kind of attacker-influenced value that could
    # carry it.
    payload = json.dumps(geojson, default=str).replace("</", "<\\/")

    return _HTML_TEMPLATE.format(
        leaflet_css=_LEAFLET_CSS,
        leaflet_js=_LEAFLET_JS,
        height=int(height),
        payload=payload,
        plate=escape(trajectory.plate_number or ""),
        fallback_lat=_FALLBACK_CENTER[0],
        fallback_lon=_FALLBACK_CENTER[1],
        fallback_zoom=_FALLBACK_ZOOM,
        satellite_default=json.dumps(bool(satellite_default)),
    )


# The document is a .format() template, so every literal CSS/JS brace is
# doubled. Placeholders: leaflet_css, leaflet_js, height, payload, plate,
# fallback_lat, fallback_lon, fallback_zoom, satellite_default.
_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="stylesheet" href="{leaflet_css}" />
<script src="{leaflet_js}"></script>
<style>
  html, body {{ margin: 0; padding: 0; }}
  #map {{
    height: {height}px; width: 100%;
    border-radius: 10px; background: #1b1f24;
  }}
  .leaflet-popup-content {{ margin: 10px 12px; min-width: 210px; }}
  .tp-popup {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 13px; }}
  .tp-popup h4 {{
    margin: 0 0 6px 0; font-size: 14px; color: #0b3d64;
    display: flex; align-items: center; gap: 6px;
  }}
  .tp-seq {{
    background: #0b3d64; color: #fff; border-radius: 50%;
    width: 20px; height: 20px; display: inline-flex;
    align-items: center; justify-content: center; font-size: 11px; flex: none;
  }}
  .tp-plate {{
    font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
    font-weight: 700; background: #ffd54f; color: #222;
    padding: 2px 8px; border-radius: 4px; letter-spacing: .5px;
    display: inline-block; margin-bottom: 6px;
  }}
  .tp-row {{ display: flex; justify-content: space-between; gap: 12px; padding: 2px 0; }}
  .tp-key {{ color: #667; }}
  .tp-val {{ font-weight: 600; color: #223; text-align: right; }}
  .tp-thumb {{
    margin-top: 8px; width: 100%; border-radius: 6px;
    border: 1px solid #dde; display: block;
  }}
  .tp-empty {{
    display: flex; align-items: center; justify-content: center;
    height: 100%; color: #9aa5b1; font-family: system-ui, sans-serif;
    font-size: 14px; text-align: center; padding: 0 24px;
  }}
  .tp-marker {{
    background: transparent; border: none;
  }}
  .tp-pin {{
    width: 26px; height: 26px; border-radius: 50%;
    border: 3px solid #fff; box-shadow: 0 1px 6px rgba(0,0,0,.5);
    color: #fff; font: 700 12px/20px system-ui, sans-serif;
    text-align: center; font-family: system-ui, sans-serif;
  }}
  .tp-arrow {{ color: #fff; text-shadow: 0 0 3px #000, 0 0 3px #000; font-size: 17px; line-height: 17px; }}
  .tp-legend {{
    background: rgba(255,255,255,.92); padding: 8px 10px; border-radius: 6px;
    font: 12px system-ui, sans-serif; line-height: 1.5; color: #223;
    box-shadow: 0 1px 5px rgba(0,0,0,.3);
  }}
  .tp-legend b {{ display: block; margin-bottom: 3px; font-size: 12px; }}
</style>
</head>
<body>
<div id="map"></div>
<script>
(function () {{
  var data = {payload};
  var plate = "{plate}";
  var mapEl = document.getElementById("map");

  var detections = (data.features || []).filter(function (f) {{
    return f.properties && f.properties.kind === "detection";
  }});
  var pathFeature = (data.features || []).filter(function (f) {{
    return f.properties && f.properties.kind === "path";
  }})[0];

  if (!detections.length) {{
    mapEl.innerHTML = '<div class="tp-empty">No mapped detections for this vehicle.<br>' +
      'The cameras that saw it have no coordinates configured.</div>';
    return;
  }}

  var map = L.map("map", {{ scrollWheelZoom: true, zoomControl: true }});

  var streets = L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }});
  var satellite = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}", {{
    maxZoom: 19,
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community'
  }});
  var labels = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}", {{
    maxZoom: 19, attribution: '&copy; Esri'
  }});

  // Satellite imagery alone has no street names, which makes a city
  // trajectory hard to place; pairing it with Esri's reference layer keeps
  // the labels without giving up the imagery.
  var hybrid = L.layerGroup([satellite, labels]);
  ({satellite_default} ? hybrid : streets).addTo(map);
  L.control.layers(
    {{ "Street map": streets, "Satellite": hybrid }},
    {{}},
    {{ collapsed: false, position: "topright" }}
  ).addTo(map);

  // Colour ramp along the path: first camera cool, last camera warm, so
  // direction of travel is readable even before the arrows are noticed.
  function rampColor(i, n) {{
    if (n <= 1) return "#2e7d32";
    var t = i / (n - 1);
    var hue = 145 - 145 * t;   // green -> red through amber
    return "hsl(" + hue.toFixed(0) + ", 78%, 44%)";
  }}

  var latlngs = detections.map(function (f) {{
    return [f.geometry.coordinates[1], f.geometry.coordinates[0]];
  }});

  if (latlngs.length > 1) {{
    // Dark casing under the coloured line: the ramp's mid-tones are low
    // contrast against satellite imagery on their own.
    L.polyline(latlngs, {{ color: "#10161c", weight: 9, opacity: .55 }}).addTo(map);
    for (var i = 0; i < latlngs.length - 1; i++) {{
      L.polyline([latlngs[i], latlngs[i + 1]], {{
        color: rampColor(i, latlngs.length - 1),
        weight: 5, opacity: .95, lineCap: "round"
      }}).addTo(map);
    }}
  }}

  function bearing(a, b) {{
    var toRad = Math.PI / 180;
    var dLon = (b[1] - a[1]) * toRad;
    var lat1 = a[0] * toRad, lat2 = b[0] * toRad;
    var y = Math.sin(dLon) * Math.cos(lat2);
    var x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
    return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
  }}

  // One arrow at each leg's midpoint, rotated to that leg's true bearing --
  // which way the vehicle travelled, not just which cameras it passed.
  for (var j = 0; j < latlngs.length - 1; j++) {{
    var a = latlngs[j], b = latlngs[j + 1];
    var mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    var deg = bearing(a, b);
    L.marker(mid, {{
      interactive: false,
      icon: L.divIcon({{
        className: "tp-marker",
        iconSize: [18, 18], iconAnchor: [9, 9],
        // The glyph points north at 0deg, so rotating by the bearing aims
        // it along the leg.
        html: '<div class="tp-arrow" style="transform: rotate(' + deg.toFixed(1) + 'deg)">&#10148;</div>'
      }})
    }}).addTo(map);
  }}

  function esc(v) {{
    if (v === null || v === undefined || v === "") return "--";
    return String(v).replace(/[&<>"']/g, function (c) {{
      return {{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }}[c];
    }});
  }}

  function fmtTime(iso) {{
    if (!iso) return "--";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return esc(iso);
    return d.toLocaleString(undefined, {{
      year: "numeric", month: "short", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit"
    }});
  }}

  function row(key, value) {{
    return '<div class="tp-row"><span class="tp-key">' + key +
           '</span><span class="tp-val">' + value + '</span></div>';
  }}

  detections.forEach(function (feature, index) {{
    var p = feature.properties;
    var latlng = [feature.geometry.coordinates[1], feature.geometry.coordinates[0]];
    var color = rampColor(index, detections.length);

    var html = '<div class="tp-popup">' +
      '<h4><span class="tp-seq">' + esc(p.sequence) + '</span>' + esc(p.camera_name) + '</h4>' +
      '<div class="tp-plate">' + esc(p.plate_number || plate) + '</div>' +
      row("Camera ID", esc(p.camera_id)) +
      row("Time", fmtTime(p.timestamp)) +
      row("Confidence", p.confidence == null ? "--" : (p.confidence * 100).toFixed(1) + "%") +
      row("Detections here", esc(p.detection_count)) +
      row("Coordinates", latlng[0].toFixed(5) + ", " + latlng[1].toFixed(5));

    if (p.vehicle_type) html += row("Vehicle", esc(p.vehicle_type));
    if (p.direction) html += row("Direction", esc(p.direction));

    var thumb = p.vehicle_image || p.plate_image;
    if (thumb) html += '<img class="tp-thumb" src="' + thumb + '" alt="Vehicle at ' + esc(p.camera_name) + '" />';
    html += '</div>';

    L.marker(latlng, {{
      title: p.camera_name + " -- " + fmtTime(p.timestamp),
      icon: L.divIcon({{
        className: "tp-marker",
        iconSize: [26, 26], iconAnchor: [13, 13], popupAnchor: [0, -14],
        html: '<div class="tp-pin" style="background:' + color + '">' + esc(p.sequence) + '</div>'
      }})
    }}).addTo(map).bindPopup(html, {{ maxWidth: 280 }});
  }});

  var legend = L.control({{ position: "bottomright" }});
  legend.onAdd = function () {{
    var div = L.DomUtil.create("div", "tp-legend");
    var summary = (data.properties || {{}});
    var parts = ["<b>" + esc(plate) + "</b>"];
    parts.push(esc(summary.cameras_visited) + " camera(s) &middot; " +
               esc(summary.mapped_points) + " mapped point(s)");
    if (summary.total_distance_km != null) {{
      parts.push("Path length: " + Number(summary.total_distance_km).toFixed(2) + " km");
    }}
    if (summary.unmapped_points) {{
      parts.push('<span style="color:#b26a00">' + esc(summary.unmapped_points) +
                 " point(s) not mapped</span>");
    }}
    div.innerHTML = parts.join("<br>");
    return div;
  }};
  legend.addTo(map);

  if (latlngs.length === 1) {{
    map.setView(latlngs[0], 15);
  }} else {{
    map.fitBounds(L.latLngBounds(latlngs), {{ padding: [45, 45] }});
  }}

  // Streamlit sizes the component iframe after the script runs, so Leaflet's
  // initial size measurement can be taken against a zero-height container
  // and render only grey tiles. Re-measuring once the frame has settled
  // fixes it; the observer covers later resizes (sidebar toggle, window).
  setTimeout(function () {{ map.invalidateSize(); }}, 200);
  if (window.ResizeObserver) {{
    new ResizeObserver(function () {{ map.invalidateSize(); }}).observe(mapEl);
  }}
}})();
</script>
</body>
</html>
"""
