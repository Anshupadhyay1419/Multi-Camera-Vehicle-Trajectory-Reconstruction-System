"""Clear all stored ALPR data: database rows, saved plate-crop images, and
the live dashboard frame. Uses the same database module as the rest of the
app, so it works correctly whether you're on SQLite or PostgreSQL.

Usage:
    python scripts/clear_data.py            # asks for confirmation
    python scripts/clear_data.py --yes      # skips confirmation (for scripts)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.utils.data_reset import clear_all_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear all stored ALPR data")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    config = load_config(args.config)

    if not args.yes:
        answer = input(
            "This deletes every stored plate-crop image, every stored "
            "event, and the current live-feed frame. Continue? [y/N] "
        )
        if answer.strip().lower() != "y":
            print("Cancelled -- nothing was deleted.")
            return

    counts = clear_all_data(config)
    print(f"Deleted {counts['events']} database row(s).")
    print(f"Deleted {counts['images']} plate-crop image(s).")
    print("Done. Restart the pipeline and dashboard to pick up the clean state.")


if __name__ == "__main__":
    main()
