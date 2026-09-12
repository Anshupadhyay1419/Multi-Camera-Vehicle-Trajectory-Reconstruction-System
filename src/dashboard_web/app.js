(() => {
  "use strict";

  const REFRESH_MS = 4000;
  const FEED_CHECK_MS = 3000;
  const HEALTH_CHECK_MS = 5000;

  const el = (id) => document.getElementById(id);

  const eventsBody = el("events-body");
  const tableTitle = el("table-title");
  const searchForm = el("search-form");
  const searchInput = el("search-input");
  const searchClear = el("search-clear");
  const searchSummary = el("search-summary");
  const autoRefreshToggle = el("auto-refresh-toggle");
  const statusDot = el("status-dot");
  const statusText = el("status-text");
  const feedImg = el("feed-img");
  const feedOffline = el("feed-offline");
  const feedUpdated = el("feed-updated");
  const detailOverlay = el("detail-overlay");
  const detailImg = el("detail-img");
  const detailFields = el("detail-fields");
  const detailClose = el("detail-close");
  const clearDataButton = el("clear-data-button");
  const cameraLabel = el("camera-label");

  let currentSearch = null; // null = showing recent events, else the query string
  let lastEvents = [];

  function fmtTime(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, {
      year: "numeric", month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  }

  // Coordinates are nullable in the database -- events recorded before
  // camera attribution existed have none, and a gate that hasn't been
  // surveyed yet has none either. Both cases render as a dash rather than
  // as "0, 0", which is a real place in the Atlantic.
  function hasCoords(ev) {
    return typeof ev.latitude === "number" && typeof ev.longitude === "number";
  }

  // 5 decimal places is ~1m at the equator -- more than enough to identify
  // a gate, and short enough to read in a table cell.
  function fmtCoords(ev) {
    if (!hasCoords(ev)) return null;
    return `${ev.latitude.toFixed(5)}, ${ev.longitude.toFixed(5)}`;
  }

  // OpenStreetMap needs no API key and loads no third-party script into
  // this page -- it's just an href the operator can click.
  function mapUrl(ev) {
    if (!hasCoords(ev)) return null;
    return `https://www.openstreetmap.org/?mlat=${ev.latitude}&mlon=${ev.longitude}#map=18/${ev.latitude}/${ev.longitude}`;
  }

  function thumbUrl(imagePath) {
    if (!imagePath) return null;
    const name = imagePath.split("/").pop();
    return `/media/${encodeURIComponent(name)}`;
  }

  function renderTable(events) {
    lastEvents = events;
    if (!events.length) {
      eventsBody.innerHTML = `<tr><td colspan="8" class="muted center">No records found.</td></tr>`;
      return;
    }

    eventsBody.innerHTML = events.map((ev, idx) => {
      const thumb = thumbUrl(ev.image_path);
      const pillClass = ev.direction === "IN" ? "pill-in" : "pill-out";
      const coords = fmtCoords(ev);
      const map = mapUrl(ev);
      return `
        <tr data-idx="${idx}">
          <td>${thumb
            ? `<img class="thumb" src="${thumb}" alt="" loading="lazy" onerror="this.replaceWith(Object.assign(document.createElement('span'),{className:'thumb thumb-missing'}))">`
            : `<span class="thumb thumb-missing"></span>`
          }</td>
          <td class="plate-text">${escapeHtml(ev.plate_number)}</td>
          <td><span class="pill ${pillClass}">${escapeHtml(ev.direction)}</span></td>
          <td>${escapeHtml(ev.vehicle_type || "—")}</td>
          <td>${escapeHtml(ev.plate_color || "—")}</td>
          <td class="camera-cell">
            <span class="camera-name">${escapeHtml(ev.camera_name || "—")}</span>
            ${ev.camera_id ? `<span class="camera-id muted">${escapeHtml(ev.camera_id)}</span>` : ""}
          </td>
          <td class="coords">${
            // stopPropagation: the whole row opens the detail overlay on
            // click, and following the map link shouldn't also do that.
            map
              ? `<a href="${map}" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation()">${escapeHtml(coords)}</a>`
              : `<span class="muted">—</span>`
          }</td>
          <td>${fmtTime(ev.timestamp)}</td>
        </tr>
      `;
    }).join("");
  }

  function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = value == null ? "" : String(value);
    return div.innerHTML;
  }

  function openDetail(idx) {
    const ev = lastEvents[idx];
    if (!ev) return;
    const thumb = thumbUrl(ev.image_path);
    detailImg.onerror = () => { detailImg.style.display = "none"; };
    detailImg.onload = () => { detailImg.style.display = ""; };
    detailImg.src = thumb || "";
    detailImg.style.display = thumb ? "" : "none";
    detailFields.innerHTML = `
      <dt>Plate</dt><dd class="plate-text">${escapeHtml(ev.plate_number)}</dd>
      <dt>Direction</dt><dd>${escapeHtml(ev.direction)}</dd>
      <dt>Vehicle type</dt><dd>${escapeHtml(ev.vehicle_type || "—")}</dd>
      <dt>Color</dt><dd>${escapeHtml(ev.plate_color || "—")}</dd>
      <dt>Series</dt><dd>${escapeHtml(ev.series_type || "—")}</dd>
      <dt>Camera ID</dt><dd>${escapeHtml(ev.camera_id || "—")}</dd>
      <dt>Camera name</dt><dd>${escapeHtml(ev.camera_name || "—")}</dd>
      <dt>Latitude</dt><dd>${hasCoords(ev) ? ev.latitude.toFixed(6) : "—"}</dd>
      <dt>Longitude</dt><dd>${hasCoords(ev) ? ev.longitude.toFixed(6) : "—"}</dd>
      ${mapUrl(ev)
        ? `<dt>Location</dt><dd><a href="${mapUrl(ev)}" target="_blank" rel="noopener noreferrer">View on map</a></dd>`
        : ""}
      <dt>Timestamp</dt><dd>${fmtTime(ev.timestamp)}</dd>
    `;
    detailOverlay.classList.remove("hidden");
  }

  eventsBody.addEventListener("click", (e) => {
    const row = e.target.closest("tr[data-idx]");
    if (row) openDetail(Number(row.dataset.idx));
  });

  detailClose.addEventListener("click", () => detailOverlay.classList.add("hidden"));
  detailOverlay.addEventListener("click", (e) => {
    if (e.target === detailOverlay) detailOverlay.classList.add("hidden");
  });

  async function fetchJson(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url} -> ${res.status}`);
    return res.json();
  }

  async function loadRecent() {
    try {
      const data = await fetchJson("/live?limit=50");
      renderTable(data.events || []);
    } catch (err) {
      eventsBody.innerHTML = `<tr><td colspan="8" class="muted center">Could not load records.</td></tr>`;
    }
  }

  async function loadSearch(query) {
    try {
      const data = await fetchJson(`/search?plate=${encodeURIComponent(query)}`);
      renderTable(data);
      searchSummary.textContent = data.length
        ? `${data.length} record${data.length === 1 ? "" : "s"} for ${query.toUpperCase()}`
        : `No records found for ${query.toUpperCase()}`;
    } catch (err) {
      eventsBody.innerHTML = `<tr><td colspan="8" class="muted center">Search failed.</td></tr>`;
      searchSummary.textContent = "";
    }
  }

  async function refreshTable() {
    if (currentSearch) {
      await loadSearch(currentSearch);
    } else {
      await loadRecent();
    }
  }

  searchForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const value = searchInput.value.trim();
    if (!value) return;
    currentSearch = value;
    tableTitle.textContent = "Search results";
    loadSearch(value);
  });

  searchClear.addEventListener("click", () => {
    currentSearch = null;
    searchInput.value = "";
    searchSummary.textContent = "";
    tableTitle.textContent = "Recent entries";
    loadRecent();
  });

  async function refreshStats() {
    try {
      const stats = await fetchJson("/stats");
      el("stat-entries").textContent = stats.entries ?? "—";
      el("stat-exits").textContent = stats.exits ?? "—";
      el("stat-unique").textContent = stats.unique_vehicles ?? "—";
      el("stat-total").textContent = stats.total_events ?? "—";
    } catch (err) {
      // Leave existing values in place rather than blanking them on a
      // transient failure.
    }
  }

  // Which gate this dashboard is showing, from the server's own
  // config.yaml. Read once at load -- it's static per device, so there's
  // nothing to poll for. Shown in the header so the page identifies its
  // camera even when the event log is empty.
  async function loadCameraLabel() {
    try {
      const settings = await fetchJson("/settings");
      const cam = settings.camera;
      if (!cam || !cam.camera_id) return;
      const coords = fmtCoords(cam);
      cameraLabel.textContent = coords
        ? `${cam.camera_name} · ${cam.camera_id} · ${coords}`
        : `${cam.camera_name} · ${cam.camera_id}`;
      cameraLabel.classList.remove("hidden");
    } catch (err) {
      // Leave the label hidden -- a missing camera block is not worth an
      // error message on an otherwise working dashboard.
    }
  }

  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        statusDot.className = "status-dot status-online";
        statusText.textContent = "Connected";
      } else {
        throw new Error("unhealthy");
      }
    } catch (err) {
      statusDot.className = "status-dot status-offline";
      statusText.textContent = "Disconnected";
    }
  }

  let feedIsLive = false;

  async function checkFeed() {
    try {
      const res = await fetch("/snapshot", { cache: "no-store" });
      if (res.ok) {
        if (!feedIsLive) {
          feedIsLive = true;
          feedOffline.classList.add("hidden");
          feedImg.classList.remove("hidden");
          feedImg.src = `/stream?_=${Date.now()}`;
        }
        feedUpdated.textContent = `Updated ${new Date().toLocaleTimeString()}`;
      } else {
        throw new Error("no snapshot");
      }
    } catch (err) {
      if (feedIsLive) {
        feedIsLive = false;
        feedImg.classList.add("hidden");
        feedOffline.classList.remove("hidden");
        feedUpdated.textContent = "—";
      }
    }
  }

  feedImg.addEventListener("error", () => {
    if (feedIsLive) {
      feedIsLive = false;
      feedImg.classList.add("hidden");
      feedOffline.classList.remove("hidden");
    }
  });

  function tick() {
    checkHealth();
    if (autoRefreshToggle.checked) {
      refreshStats();
      refreshTable();
    }
  }

  clearDataButton.addEventListener("click", async () => {
    const confirmed = window.confirm(
      "This permanently deletes every stored event, every plate-crop image, "
      + "and the current live-feed frame. This cannot be undone. Continue?"
    );
    if (!confirmed) return;

    clearDataButton.disabled = true;
    clearDataButton.textContent = "Clearing…";
    try {
      const res = await fetch("/clear", { method: "POST" });
      if (!res.ok) throw new Error(`/clear -> ${res.status}`);
      currentSearch = null;
      searchInput.value = "";
      searchSummary.textContent = "";
      tableTitle.textContent = "Recent entries";
      await Promise.all([refreshStats(), refreshTable()]);
    } catch (err) {
      window.alert("Failed to clear data. Check the dashboard server is running and try again.");
    } finally {
      clearDataButton.disabled = false;
      clearDataButton.textContent = "Clear all data";
    }
  });

  // Initial load
  loadCameraLabel();
  checkHealth();
  checkFeed();
  refreshStats();
  refreshTable();

  setInterval(tick, REFRESH_MS);
  setInterval(checkFeed, FEED_CHECK_MS);
})();
