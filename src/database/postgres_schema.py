"""
Production-grade PostgreSQL database schema for ALPR University Gate system.

This replaces SQLite with PostgreSQL for:
- High concurrent connections
- Better indexing for plate search
- JSONB storage for metadata
- Full-text search support
- Historical data retention
- Audit logging

Tables:
  - vehicles: Unique vehicle records (deduplicated by plate)
  - entries: Individual IN/OUT events
  - images: Plate crop images with quality metadata
  - logs: System logs, errors, OCR attempts (audit trail)
  - cameras: Camera metadata and health checks
"""

CREATE_TABLES = """
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS uuid-ossp;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Cameras table: metadata about surveillance cameras
CREATE TABLE IF NOT EXISTS cameras (
    camera_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    location VARCHAR(255),
    ip_address INET,
    resolution VARCHAR(50),  -- "1920x1080"
    fps INTEGER,  -- frames per second
    installed_date TIMESTAMP DEFAULT NOW(),
    last_health_check TIMESTAMP,
    status VARCHAR(50) DEFAULT 'UNKNOWN',  -- 'ACTIVE', 'MAINTENANCE', 'INACTIVE'
    metadata JSONB,  -- WDR setting, IR mode, calibration data, etc.
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cameras_status ON cameras(status);
CREATE INDEX IF NOT EXISTS idx_cameras_ip ON cameras(ip_address);

-- Vehicles table: unique vehicles (deduplicated by plate)
CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id BIGSERIAL PRIMARY KEY,
    plate_number VARCHAR(20) NOT NULL UNIQUE,
    plate_series VARCHAR(50),  -- "normal", "commercial", "temporary", "temporary_taxi"
    plate_color VARCHAR(50),  -- "White", "Yellow"
    first_seen TIMESTAMP NOT NULL,
    last_seen TIMESTAMP NOT NULL,
    total_entries INTEGER DEFAULT 0,  -- count of IN events
    total_exits INTEGER DEFAULT 0,    -- count of OUT events
    current_status VARCHAR(50) DEFAULT 'UNKNOWN',  -- 'INSIDE', 'OUTSIDE', 'UNKNOWN'
    vehicle_type VARCHAR(100),  -- inferred: "Car", "SUV", "Commercial", "Auto", etc.
    vehicle_color VARCHAR(100),  -- observed color
    owner_contact VARCHAR(255),  -- optional: scanned from registration
    metadata JSONB,  -- observed colors, frequent entry times, etc.
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vehicles_plate ON vehicles(plate_number);
CREATE INDEX IF NOT EXISTS idx_vehicles_timestamp ON vehicles(first_seen DESC);
CREATE INDEX IF NOT EXISTS idx_vehicles_status ON vehicles(current_status);
CREATE INDEX IF NOT EXISTS idx_vehicles_series ON vehicles(plate_series);
CREATE INDEX IF NOT EXISTS idx_vehicles_plate_trgm ON vehicles USING GIN (plate_number gin_trgm_ops);

-- Entries table: individual IN/OUT events (temporal log)
CREATE TABLE IF NOT EXISTS entries (
    entry_id BIGSERIAL PRIMARY KEY,
    vehicle_id BIGINT NOT NULL REFERENCES vehicles(vehicle_id) ON DELETE CASCADE,
    camera_id INTEGER REFERENCES cameras(camera_id),
    plate_number VARCHAR(20),  -- cached for quick lookup without JOIN
    direction VARCHAR(10) NOT NULL,  -- "IN" or "OUT"
    confidence NUMERIC(5, 4),  -- overall confidence 0.0-1.0
    ocr_text_raw VARCHAR(50),  -- raw OCR output before validation
    ocr_confidence NUMERIC(5, 4),  -- PaddleOCR confidence
    image_path VARCHAR(255),  -- relative path to plate crop
    image_quality NUMERIC(5, 4),  -- blur score 0-1
    vehicle_type VARCHAR(100),
    vehicle_color VARCHAR(100),
    plate_color VARCHAR(50),
    frame_number INTEGER,  -- video frame for debugging
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB,  -- centroid, bounding box, track_id, etc.
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entries_vehicle ON entries(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_entries_plate ON entries(plate_number);
CREATE INDEX IF NOT EXISTS idx_entries_timestamp ON entries(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_entries_direction ON entries(direction);
CREATE INDEX IF NOT EXISTS idx_entries_camera ON entries(camera_id);
CREATE INDEX IF NOT EXISTS idx_entries_confidence ON entries(confidence DESC);
-- Composite index for common queries: "give me all OUT events in past hour"
CREATE INDEX IF NOT EXISTS idx_entries_direction_timestamp ON entries(direction, timestamp DESC);

-- Images table: plate crops with quality metadata
CREATE TABLE IF NOT EXISTS images (
    image_id BIGSERIAL PRIMARY KEY,
    entry_id BIGINT NOT NULL REFERENCES entries(entry_id) ON DELETE CASCADE,
    image_path VARCHAR(255) NOT NULL,
    image_size_bytes INTEGER,
    width INTEGER,
    height INTEGER,
    blur_score NUMERIC(5, 4),
    contrast NUMERIC(5, 4),
    brightness NUMERIC(5, 4),
    quality_grade VARCHAR(10),  -- "A", "B", "C"
    enhancement_applied VARCHAR(100),  -- "clahe", "denoise", "sr", etc.
    metadata JSONB,  -- histogram, mean intensity, etc.
    stored_at TIMESTAMP DEFAULT NOW(),
    archived_at TIMESTAMP,  -- NULL if active, timestamp if moved to cold storage
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_images_entry ON images(entry_id);
CREATE INDEX IF NOT EXISTS idx_images_quality ON images(quality_grade);
CREATE INDEX IF NOT EXISTS idx_images_archived ON images(archived_at);

-- Logs table: system operational logs (audit trail)
CREATE TABLE IF NOT EXISTS logs (
    log_id BIGSERIAL PRIMARY KEY,
    vehicle_id BIGINT REFERENCES vehicles(vehicle_id) ON DELETE SET NULL,
    entry_id BIGINT REFERENCES entries(entry_id) ON DELETE SET NULL,
    log_level VARCHAR(20),  -- "INFO", "WARNING", "ERROR", "DEBUG"
    module VARCHAR(100),  -- "vehicle_detector", "ocr", "fusion", "database", etc.
    message TEXT,
    error_details TEXT,  -- traceback if applicable
    ocr_attempt JSONB,  -- {text, confidence, raw_text, engines_tried, correction_applied}
    validation_result JSONB,  -- {is_valid, plate_series, error_msg}
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(log_level);
CREATE INDEX IF NOT EXISTS idx_logs_module ON logs(module);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_logs_vehicle ON logs(vehicle_id);
-- Full-text search on error messages
CREATE INDEX IF NOT EXISTS idx_logs_message_trgm ON logs USING GIN (message gin_trgm_ops);

-- Duplicate detection log (for auditing duplicate filter behavior)
CREATE TABLE IF NOT EXISTS duplicates_detected (
    duplicate_id BIGSERIAL PRIMARY KEY,
    vehicle_id BIGINT NOT NULL REFERENCES vehicles(vehicle_id),
    entry_attempt_id BIGINT REFERENCES entries(entry_id),
    duplicate_reason VARCHAR(100),  -- "same_track_id", "plate_match", "time_window"
    time_since_last_entry INTERVAL,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_duplicates_vehicle ON duplicates_detected(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_duplicates_timestamp ON duplicates_detected(timestamp DESC);

-- Analytics materialized view (for dashboard)
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_traffic_stats AS
SELECT
    DATE(e.timestamp) as date,
    c.camera_id,
    c.name as camera_name,
    COUNT(CASE WHEN e.direction = 'IN' THEN 1 END) as entries_count,
    COUNT(CASE WHEN e.direction = 'OUT' THEN 1 END) as exits_count,
    ROUND(AVG(e.ocr_confidence)::NUMERIC, 4) as avg_ocr_confidence,
    COUNT(DISTINCT e.vehicle_id) as unique_vehicles
FROM entries e
LEFT JOIN cameras c ON e.camera_id = c.camera_id
GROUP BY DATE(e.timestamp), c.camera_id, c.name;

CREATE INDEX IF NOT EXISTS idx_daily_traffic_date ON daily_traffic_stats(date DESC);

"""

# Migrations
MIGRATION_V2_JSONB_SUPPORT = """
-- Add JSONB fields for better extensibility
ALTER TABLE vehicles ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';
ALTER TABLE entries ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';
ALTER TABLE images ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';
ALTER TABLE logs ADD COLUMN IF NOT EXISTS ocr_attempt JSONB;
ALTER TABLE logs ADD COLUMN IF NOT EXISTS validation_result JSONB;
"""

MIGRATION_V3_ARCHIVE_SUPPORT = """
-- Add archival columns for cold storage
ALTER TABLE images ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP;
ALTER TABLE entries ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP;

-- Create archive table for historical data
CREATE TABLE IF NOT EXISTS entries_archive (LIKE entries INCLUDING ALL);
CREATE TABLE IF NOT EXISTS images_archive (LIKE images INCLUDING ALL);
"""

MIGRATION_V4_ADD_CAMERA_METADATA = """
-- Add camera health monitoring
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS last_health_check TIMESTAMP;
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'UNKNOWN';
"""


class MigrationManager:
    """Manage database migrations from SQLite to PostgreSQL."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.version = "1.0.0"

    def create_schema(self) -> None:
        """Create all tables from scratch."""
        from sqlalchemy import text, create_engine

        engine = create_engine(self.db_url)
        with engine.connect() as conn:
            for statement in CREATE_TABLES.split(";"):
                if statement.strip():
                    conn.execute(text(statement))
            conn.commit()

        print("✓ PostgreSQL schema created")

    def migrate_v2_jsonb(self) -> None:
        """Add JSONB support."""
        from sqlalchemy import text, create_engine

        engine = create_engine(self.db_url)
        with engine.connect() as conn:
            for statement in MIGRATION_V2_JSONB_SUPPORT.split(";"):
                if statement.strip():
                    conn.execute(text(statement))
            conn.commit()

        print("✓ Migration v2 (JSONB support) applied")

    def migrate_sqlite_to_postgres(self, sqlite_path: str) -> None:
        """Migrate data from SQLite to PostgreSQL.

        Args:
            sqlite_path: Path to existing SQLite database
        """
        import sqlite3
        from sqlalchemy import text, create_engine, insert
        from src.database.models import VehicleEvent

        sqlite_conn = sqlite3.connect(sqlite_path)
        sqlite_cursor = sqlite_conn.cursor()

        postgres_engine = create_engine(self.db_url)

        # Get all events from SQLite
        sqlite_cursor.execute("SELECT * FROM vehicle_events")
        columns = [desc[0] for desc in sqlite_cursor.description]
        rows = sqlite_cursor.fetchall()

        print(f"Migrating {len(rows)} events from SQLite to PostgreSQL...")

        with postgres_engine.begin() as conn:
            for row in rows:
                data = dict(zip(columns, row))
                conn.execute(insert(VehicleEvent).values(**data))

        sqlite_conn.close()
        print(f"✓ Migrated {len(rows)} events")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PostgreSQL schema setup")
    parser.add_argument("--db-url", required=True, help="PostgreSQL connection URL")
    parser.add_argument("--create-schema", action="store_true", help="Create schema from scratch")
    parser.add_argument("--migrate-sqlite", help="Migrate from SQLite database path")

    args = parser.parse_args()

    manager = MigrationManager(args.db_url)

    if args.create_schema:
        manager.create_schema()
        manager.migrate_v2_jsonb()

    if args.migrate_sqlite:
        manager.migrate_sqlite_to_postgres(args.migrate_sqlite)
