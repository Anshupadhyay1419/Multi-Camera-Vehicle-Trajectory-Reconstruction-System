"""
Enhanced Streamlit dashboard for ALPR University Gate system.

Features:
  - Live feed with real-time updates
  - Advanced search and filtering (plate, date range, direction, vehicle type)
  - Daily traffic statistics and analytics
  - Vehicle history tracking
  - Image gallery with quality assessment
  - System health monitoring
  - Export functionality (CSV, PDF)

Run with:
  streamlit run src/dashboard/app.py
  streamlit run src/dashboard/app.py --config config/streamlit.toml
"""

from __future__ import annotations

import sys
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import load_config
from pathlib import Path
from src.database import db as database


# ── Page Configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="ALPR University Gate",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .plate-badge {
        background-color: #FFD700;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        font-family: monospace;
    }
    .status-in { color: #28a745; }
    .status-out { color: #dc3545; }
    </style>
""", unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────────────
@st.cache_resource
def init_db_connection():
    """Initialize database connection."""
    try:
        repo_root = Path(__file__).resolve().parents[2]
        config_path = str(repo_root / "config" / "config.yaml")
        config = load_config(config_path)
        db_path = config.get("database", {}).get("path", "data/alpr.db")
        database.init_db(db_path)
        return True
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return False


@st.cache_data(ttl=5)
def load_config_cached():
    """Load and cache configuration."""
    try:
        repo_root = Path(__file__).resolve().parents[2]
        config_path = str(repo_root / "config" / "config.yaml")
        return load_config(config_path)
    except Exception as e:
        st.error(f"Config load failed: {e}")
        return {}


# ── Data Loading Functions ────────────────────────────────────────────
def load_recent_events(limit: int = 100) -> list[dict]:
    """Load recent events from database."""
    try:
        with database.get_session() as session:
            return database.get_all_events(session, limit=limit)
    except Exception as exc:
        st.error(f"Failed to load events: {exc}")
        return []


def load_direction_events(direction: str, limit: int = 50) -> list[dict]:
    """Load events filtered by direction."""
    try:
        with database.get_session() as session:
            return database.get_events_by_direction(session, direction, limit=limit)
    except Exception as exc:
        st.error(f"Failed to load {direction} events: {exc}")
        return []


def search_plate_history(plate: str) -> list[dict]:
    """Search all events for a specific plate."""
    try:
        with database.get_session() as session:
            return database.search_events(session, plate_number=plate.upper())
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return []


def get_daily_stats() -> dict:
    """Get today's traffic statistics."""
    try:
        with database.get_session() as session:
            return database.get_daily_stats(session)
    except Exception as exc:
        st.error(f"Failed to get stats: {exc}")
        return {}


def prepare_dataframe(events: list[dict]) -> pd.DataFrame:
    """Convert events list to DataFrame for analysis."""
    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _html_table(df) -> None:
    """Render a DataFrame as plain HTML instead of st.dataframe.

    st.dataframe serialises through Apache Arrow, and this deployment pairs
    pyarrow 25 with numpy 1.26 -- an ABI mismatch (pyarrow 25 is built
    against numpy 2.x) whose conversion SEGFAULTS under some conditions.
    That is not a catchable exception: it kills the Streamlit server, so the
    user sees "Connection error" rather than an error message.

    numpy cannot simply be upgraded here -- requirements.txt pins 1.26.4 and
    torch, ultralytics and OpenCV are built against it -- so the tables avoid
    Arrow instead. Same approach as the multi-camera dashboard's _table().
    """
    from html import escape

    if df is None or len(df) == 0:
        st.caption("Nothing to show.")
        return

    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{escape('--' if pd.isna(v) else str(v))}</td>" for v in row
        ) + "</tr>"
        for row in df.itertuples(index=False, name=None)
    )
    st.markdown(
        '<div style="overflow-x:auto"><table style="border-collapse:collapse;'
        'width:100%;font-size:.85rem">'
        f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>',
        unsafe_allow_html=True,
    )


# ── Dashboard Main ────────────────────────────────────────────────────
def main():
    # Initialize
    if not init_db_connection():
        st.stop()

    config = load_config_cached()
    
    st.title("🚗 ALPR University Gate — Live Monitor")
    st.markdown("Real-time license plate recognition and vehicle tracking system")

    # ── Sidebar Controls ──────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Controls")

        # Refresh settings
        auto_refresh = st.checkbox(
            "Auto-refresh (5 sec)", value=True, key="auto_refresh"
        )
        refresh_interval = st.slider("Refresh interval (sec)", 2, 30, 5)

        st.divider()

        # Search
        st.subheader("🔍 Search")
        search_plate = st.text_input("Search plate number", "", placeholder="e.g., DL01AB1234").strip().upper()

        st.divider()

        # Filters
        st.subheader("🎯 Filters")
        filter_direction = st.multiselect("Direction", ["IN", "OUT"], default=["IN", "OUT"])
        filter_days = st.slider("Last N days", 1, 30, 7)

        st.divider()

        # Statistics
        st.subheader("📊 System Health")
        if st.button("Refresh Stats"):
            st.rerun()

    # ── Main Content ─────────────────────────────────────────────────
    tab_live, tab_search, tab_analytics, tab_history, tab_settings = st.tabs(
        ["📋 Live Feed", "🔍 Search", "📊 Analytics", "🚗 Vehicle History", "⚙️ Settings"]
    )

    # ──────────────────────── TAB: Live Feed ──────────────────────────
    with tab_live:
        st.subheader("📋 Recent Events (Real-Time)")

        # Load latest events
        events = load_recent_events(limit=100)
        df = prepare_dataframe(events)

        if events:
            # Quick statistics
            col1, col2, col3, col4 = st.columns(4)

            in_events = sum(1 for e in events if e.get("direction") == "IN")
            out_events = sum(1 for e in events if e.get("direction") == "OUT")
            unique_plates = len(set(e.get("plate_number", "") for e in events))

            with col1:
                st.metric("📥 Entries (today)", in_events)
            with col2:
                st.metric("📤 Exits (today)", out_events)
            with col3:
                st.metric("🚗 Unique Vehicles", unique_plates)
            with col4:
                st.metric("📍 Total Events", len(events))

            st.divider()

            # Events table
            st.write(f"**Showing {len(events)} most recent events**")

            for idx, event in enumerate(events[:20]):  # Show top 20
                with st.container():
                    col_img, col_info = st.columns([1, 3])

                    # Image preview
                    with col_img:
                        # Vehicle and plate thumbnails when the detection has
                        # them; otherwise the plate crop, as before.
                        vehicle_thumb = event.get("vehicle_thumbnail_path") or ""
                        plate_thumb = event.get("plate_thumbnail_path") or ""
                        img_path = event.get("image_path", "")
                        shown = False
                        if vehicle_thumb and Path(vehicle_thumb).exists():
                            st.image(vehicle_thumb, width=120)
                            shown = True
                        plate_img = plate_thumb if plate_thumb and Path(plate_thumb).exists() else img_path
                        if plate_img and Path(plate_img).exists():
                            st.image(plate_img, width=120)
                            shown = True
                        if not shown:
                            st.markdown("📷 *No image*")

                    # Event details
                    with col_info:
                        plate = event.get("plate_number", "N/A")
                        direction = event.get("direction", "N/A")
                        timestamp = event.get("timestamp", "N/A")
                        vehicle_type = event.get("vehicle_type", "N/A")
                        color = event.get("plate_color", "N/A")
                        series = event.get("series_type", "N/A")
                        camera_name = event.get("camera_name") or "N/A"
                        camera_id = event.get("camera_id") or "N/A"
                        latitude = event.get("latitude")
                        longitude = event.get("longitude")

                        # Plate in badge
                        st.markdown(
                            f'<span class="plate-badge">{plate}</span>',
                            unsafe_allow_html=True,
                        )

                        # Direction badge
                        direction_class = "status-in" if direction == "IN" else "status-out"
                        direction_emoji = "📥" if direction == "IN" else "📤"

                        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                        with col_d1:
                            st.markdown(f'<span class="{direction_class}">{direction_emoji} {direction}</span>', unsafe_allow_html=True)
                        vehicle_class = event.get("vehicle_class")
                        vehicle_color = event.get("vehicle_color")
                        with col_d2:
                            # YOLO class when known, keeping the registration
                            # category alongside; unchanged for older rows.
                            st.write(f"🚗 {vehicle_class.title()} ({vehicle_type})"
                                     if vehicle_class else f"🚗 {vehicle_type}")
                        with col_d3:
                            st.write(f"🎨 {vehicle_color}" if vehicle_color else f"🎨 {color}")
                        with col_d4:
                            st.write(f"📌 {series} · plate {color}" if vehicle_color else f"📌 {series}")

                        # Coordinates are nullable (pre-camera-metadata rows,
                        # and gates that haven't been surveyed), so only
                        # render them when they're actually there.
                        if latitude is not None and longitude is not None:
                            location = f" · 📍 {latitude:.5f}, {longitude:.5f}"
                        else:
                            location = ""
                        st.caption(
                            f"⏰ {timestamp} · 🎥 {camera_name} ({camera_id}){location}"
                        )

                    st.divider()

            if auto_refresh:
                time.sleep(refresh_interval)
                st.rerun()
        else:
            st.info("No events yet. Waiting for vehicle detections...")

    # ──────────────────────── TAB: Search ─────────────────────────────
    with tab_search:
        st.subheader("🔍 Search by Plate Number")

        if search_plate:
            results = search_plate_history(search_plate)

            if results:
                st.success(f"Found **{len(results)}** events for `{search_plate}`")

                df_results = prepare_dataframe(results)

                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Events", len(results))
                with col2:
                    entries = sum(1 for r in results if r.get("direction") == "IN")
                    st.metric("Entries", entries)
                with col3:
                    exits = sum(1 for r in results if r.get("direction") == "OUT")
                    st.metric("Exits", exits)

                st.divider()

                # Timeline
                st.write("**Event Timeline**")
                if not df_results.empty and "timestamp" in df_results.columns:
                    fig = px.timeline(
                        df_results,
                        x_start="timestamp",
                        x_end="timestamp",
                        y="direction",
                        color="direction",
                        title=f"Events for {search_plate}",
                        labels={"direction": "Direction"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Detailed table
                st.write("**Event Details**")
                display_cols = ["plate_number", "direction", "vehicle_type", "plate_color",
                                "camera_id", "camera_name", "latitude", "longitude", "timestamp"]
                df_display = df_results[[col for col in display_cols if col in df_results.columns]]
                _html_table(df_display)
            else:
                st.warning(f"No events found for plate `{search_plate}`")
        else:
            st.info("Enter a plate number to search for vehicle history")

    # ──────────────────────── TAB: Analytics ──────────────────────────
    with tab_analytics:
        st.subheader("📊 Traffic Analytics")

        # Load all events for analysis
        all_events = load_recent_events(limit=500)

        if all_events:
            df_all = prepare_dataframe(all_events)

            # Traffic by direction
            if "direction" in df_all.columns:
                col1, col2 = st.columns(2)

                with col1:
                    direction_counts = df_all["direction"].value_counts()
                    fig_direction = px.pie(
                        values=direction_counts.values,
                        names=direction_counts.index,
                        title="Traffic by Direction",
                        color_discrete_map={"IN": "#28a745", "OUT": "#dc3545"},
                    )
                    st.plotly_chart(fig_direction, use_container_width=True)

                with col2:
                    if "vehicle_type" in df_all.columns:
                        vehicle_counts = df_all["vehicle_type"].value_counts().head(10)
                        fig_vehicle = px.bar(
                            x=vehicle_counts.index,
                            y=vehicle_counts.values,
                            title="Top Vehicle Types",
                            labels={"x": "Vehicle Type", "y": "Count"},
                        )
                        st.plotly_chart(fig_vehicle, use_container_width=True)

            # Events over time
            if "timestamp" in df_all.columns:
                df_all["hour"] = pd.to_datetime(df_all["timestamp"]).dt.hour
                hourly_counts = df_all.groupby("hour").size()

                fig_timeline = px.bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    title="Events by Hour",
                    labels={"x": "Hour", "y": "Event Count"},
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

            # Plate color distribution
            if "plate_color" in df_all.columns:
                color_counts = df_all["plate_color"].value_counts()
                fig_color = px.pie(
                    values=color_counts.values,
                    names=color_counts.index,
                    title="Plate Colors",
                )
                st.plotly_chart(fig_color, use_container_width=True)
        else:
            st.info("No data available for analysis")

    # ──────────────────────── TAB: Vehicle History ────────────────────
    with tab_history:
        st.subheader("🚗 Vehicle History")

        plate_input = st.text_input("Enter plate number", "", placeholder="DL01AB1234", key="history_plate").upper()

        if plate_input:
            history = search_plate_history(plate_input)

            if history:
                st.success(f"**{len(history)}** records found for **{plate_input}**")

                df_history = prepare_dataframe(history)

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                first_seen = df_history["timestamp"].min() if "timestamp" in df_history.columns else None
                last_seen = df_history["timestamp"].max() if "timestamp" in df_history.columns else None
                entries_count = sum(1 for h in history if h.get("direction") == "IN")
                exits_count = sum(1 for h in history if h.get("direction") == "OUT")

                with col1:
                    st.metric("First Seen", first_seen.strftime("%Y-%m-%d %H:%M") if first_seen else "N/A")
                with col2:
                    st.metric("Last Seen", last_seen.strftime("%Y-%m-%d %H:%M") if last_seen else "N/A")
                with col3:
                    st.metric("Entries", entries_count)
                with col4:
                    st.metric("Exits", exits_count)

                st.divider()

                # Vehicle info
                vehicle_info = history[0] if history else {}
                st.write("**Vehicle Information**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if vehicle_info.get("vehicle_class"):
                        st.write(f"**Type:** {vehicle_info['vehicle_class'].title()} "
                                 f"({vehicle_info.get('vehicle_type', 'N/A')})")
                    else:
                        st.write(f"**Type:** {vehicle_info.get('vehicle_type', 'N/A')}")
                with col2:
                    if vehicle_info.get("vehicle_color"):
                        st.write(f"**Color:** {vehicle_info['vehicle_color']} "
                                 f"(plate {vehicle_info.get('plate_color', 'N/A')})")
                    else:
                        st.write(f"**Plate Color:** {vehicle_info.get('plate_color', 'N/A')}")
                with col3:
                    st.write(f"**Series:** {vehicle_info.get('series_type', 'N/A')}")
                with col4:
                    st.write(f"**Total Events:** {len(history)}")

                st.write(
                    f"**Last seen at:** {vehicle_info.get('camera_name') or 'N/A'} "
                    f"({vehicle_info.get('camera_id') or 'N/A'})"
                )

                st.divider()

                # Events table
                st.write("**Complete Event Log**")
                display_cols = ["direction", "vehicle_type", "camera_id", "camera_name",
                                "latitude", "longitude", "timestamp"]
                present_cols = [col for col in display_cols if col in df_history.columns]
                if present_cols:
                    _html_table(df_history[present_cols])
            else:
                st.warning(f"No history found for {plate_input}")
        else:
            st.info("Enter a plate number to view vehicle history")

    # ──────────────────────── TAB: Settings ───────────────────────────
    with tab_settings:
        st.subheader("⚙️ System Settings & Configuration")

        # Display current configuration
        st.write("**Database Configuration**")
        db_type = database.get_db_type() if hasattr(database, 'get_db_type') else "unknown"
        st.info(f"Database Type: {db_type}")

        st.write("**Detection Settings**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Vehicle Confidence: {config.get('detection', {}).get('vehicle_confidence', 'N/A')}")
        with col2:
            st.write(f"Plate Confidence: {config.get('detection', {}).get('plate_confidence', 'N/A')}")

        st.write("**OCR Configuration**")
        ocr_backend = config.get('ocr', {}).get('backend', 'unknown')
        st.write(f"OCR Backend: **{ocr_backend}**")

        st.write("**Tracking**")
        st.write(f"Lost Track Timeout: {config.get('tracking', {}).get('lost_track_timeout', 'N/A')} frames")

        # Export options
        st.divider()
        st.subheader("📥 Export")

        events = load_recent_events(limit=1000)
        if events:
            df_export = prepare_dataframe(events)

            # CSV export
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"alpr_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

