"""
Filter license plate detections from CSV.

Groups temporally close detections into a single vehicle pass and keeps the
highest-confidence plate per pass. This removes one-off OCR glitches that can
otherwise look like valid unique plates.
"""

import pandas as pd
import sys
from pathlib import Path

def filter_unique_detections(
    csv_path,
    output_path=None,
    max_gap_seconds: float = 0.75,
    max_gap_frames: int = 20,
):
    """
    Read CSV detections and keep only unique plates with highest OCR confidence.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to save filtered CSV (if None, uses same name with _unique suffix)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    print(f"Total detections: {len(df)}")
    print(f"Unique plates: {df['plate_number'].nunique()}")
    
    # Convert confidence strings to float
    df['ocr_confidence'] = df['ocr_confidence'].astype(float)
    
    # Sort detections in time order and cluster nearby rows that likely belong
    # to the same vehicle pass.
    sort_cols = [col for col in ['timestamp', 'frame'] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    clusters = []
    current = []

    def _close_enough(prev_row, row) -> bool:
        time_gap_ok = False
        frame_gap_ok = False

        if 'timestamp' in df.columns:
            try:
                prev_ts = float(prev_row['timestamp'])
                row_ts = float(row['timestamp'])
                time_gap_ok = (row_ts - prev_ts) <= max_gap_seconds
            except Exception:
                time_gap_ok = False

        if 'frame' in df.columns:
            try:
                frame_gap_ok = (int(row['frame']) - int(prev_row['frame'])) <= max_gap_frames
            except Exception:
                frame_gap_ok = False

        return time_gap_ok or frame_gap_ok

    for _, row in df.iterrows():
        if not current:
            current.append(row)
            continue

        if _close_enough(current[-1], row):
            current.append(row)
        else:
            clusters.append(current)
            current = [row]

    if current:
        clusters.append(current)

    # Keep the highest-confidence row from each temporal cluster.
    unique_rows = []
    for cluster in clusters:
        cluster_df = pd.DataFrame(cluster)
        best_row = cluster_df.loc[cluster_df['ocr_confidence'].astype(float).idxmax()]
        unique_rows.append(best_row)

    df_unique = pd.DataFrame(unique_rows)
    
    # Sort by timestamp
    df_unique = df_unique.sort_values('timestamp').reset_index(drop=True)
    
    # Determine output path
    if output_path is None:
        output_path = Path(csv_path).with_stem(Path(csv_path).stem + '_unique')
    
    # Save filtered CSV
    df_unique.to_csv(output_path, index=False)
    
    print(f"\nFiltered CSV saved to: {output_path}")
    print(f"Unique detections: {len(df_unique)}")
    print("\nTop detections:")
    print(df_unique[['frame', 'timestamp', 'plate_number', 'ocr_confidence']].to_string())
    
    return df_unique

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/filter_detections.py <csv_file> [output_csv]")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    filter_unique_detections(csv_file, output_file)
