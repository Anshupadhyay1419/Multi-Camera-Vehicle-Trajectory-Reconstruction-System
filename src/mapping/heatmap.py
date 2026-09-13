"""
Traffic heatmap: detection volume per camera site, on a 2D map.

Each camera contributes one weighted point at its surveyed coordinates; the
weight is how many vehicles it recorded in the chosen scope. Leaflet.heat
blends nearby points, so neighbouring busy sites read as one hot area. Every
site also gets a labelled circle sized by its count, so the numbers stay
readable and a site with no traffic is still visible (grey).
"""

from __future__ import annotations

import json
from html import escape

_LEAFLET_CSS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css"
_LEAFLET_JS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"
_HEAT_JS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet.heat/0.2.0/leaflet-heat.js"


def render_traffic_heatmap(sites: list[dict], height: int = 460) -> str:
    """Render a heatmap document.

    Args:
        sites: dicts with camera_id, camera_name, latitude, longitude, detections.
               Sites without coordinates are skipped (they cannot be placed).
    """
    points = [
        {"id": s["camera_id"], "name": s["camera_name"], "lat": float(s["latitude"]),
         "lon": float(s["longitude"]), "count": int(s.get("detections") or 0)}
        for s in sites if s.get("latitude") is not None and s.get("longitude") is not None
    ]
    payload = json.dumps(points).replace("</", "<\\/")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="{_LEAFLET_CSS}"><script src="{_LEAFLET_JS}"></script>
<script src="{_HEAT_JS}"></script>
<style>html,body{{margin:0}}#map{{height:{int(height)}px;border-radius:10px}}
.lbl{{background:rgba(255,255,255,.9);border:0;box-shadow:0 1px 4px rgba(0,0,0,.3);
font:600 12px system-ui,sans-serif;padding:2px 6px;border-radius:4px}}</style></head>
<body><div id="map"></div><script>
(function () {{
  var pts = {payload};
  var map = L.map("map");
  var streets = L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",
    {{maxZoom: 19, attribution: "&copy; OpenStreetMap contributors"}}).addTo(map);
  var satellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",
    {{maxZoom: 19, attribution: "Tiles &copy; Esri"}});
  L.control.layers({{"Street map": streets, "Satellite": satellite}}, {{}}, {{collapsed: false}}).addTo(map);
  if (!pts.length) {{ map.setView([28.6304, 77.2177], 12); return; }}
  var max = Math.max.apply(null, pts.map(function (p) {{ return p.count; }})) || 1;
  if (L.heatLayer) {{
    L.heatLayer(pts.filter(function (p) {{ return p.count > 0; }})
      .map(function (p) {{ return [p.lat, p.lon, p.count / max]; }}),
      {{radius: 55, blur: 40, maxZoom: 14, max: 1.0}}).addTo(map);
  }}
  pts.forEach(function (p) {{
    var share = p.count / max;
    var colour = p.count ? "hsl(" + (120 - 120 * share).toFixed(0) + ",85%,45%)" : "#9ca3af";
    L.circleMarker([p.lat, p.lon], {{radius: 8 + 14 * share, color: "#fff", weight: 2,
      fillColor: colour, fillOpacity: .85}}).addTo(map)
      .bindTooltip(p.name + ": " + p.count + " vehicle(s)", {{permanent: true, direction: "top",
        className: "lbl", offset: [0, -10]}});
  }});
  map.fitBounds(L.latLngBounds(pts.map(function (p) {{ return [p.lat, p.lon]; }})),
                {{padding: [60, 60], maxZoom: 15}});
  setTimeout(function () {{ map.invalidateSize(); }}, 200);
}})();
</script></body></html>"""
