#!/usr/bin/env python
"""
Batch preprocess MIDI files into conductor tokens using metadata CSV.

Usage:
    python scripts/preprocess.py --data_root data/raw --output data/processed/tokens --metadata data/processed/cleaned_metadata.csv
"""
import argparse
import sys
import datetime
from pathlib import Path

try:
    import pandas as pd
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}. Please install pandas and tqdm.")
    sys.exit(1)

# Make preprocessor modules importable
ROOT = Path(__file__).resolve().parents[1]
PREPROC_PATH = ROOT / "src" / "preprocessor"
if str(PREPROC_PATH) not in sys.path:
    sys.path.insert(0, str(PREPROC_PATH))

from dataset_builder import DatasetBuilder  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Batch preprocess MIDI files into tokens using metadata")
    parser.add_argument(
        "--data_root",
        required=True,
        help="Root directory where MIDI files are located (joined with relative paths in CSV)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--metadata",
        default="data/processed/cleaned_metadata.csv",
        help="Path to metadata CSV file (default: data/processed/cleaned_metadata.csv)",
    )
    parser.add_argument(
        "--error_log",
        default="preprocess_errors.log",
        help="Path to error log file (default: preprocess_errors.log)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N files (for quick tests)",
    )
    return parser.parse_args()


def log_error(log_path: Path, midi_path: str, error_msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {midi_path} | ERROR: {error_msg}\n")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata)
    error_log_path = Path(args.error_log)

    if not data_root.exists():
        print(f"[ERROR] Data root path does not exist: {data_root}")
        sys.exit(1)
    
    if not metadata_path.exists():
        print(f"[ERROR] Metadata CSV not found: {metadata_path}")
        sys.exit(1)

    # Reset error log
    with open(error_log_path, "w", encoding="utf-8") as f:
        f.write(f"Preprocess Error Log - Started at {datetime.datetime.now()}\n")
        f.write("="*60 + "\n")

    print(f"[INFO] Loading metadata from {metadata_path}...")
    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        sys.exit(1)

    if args.limit:
        df = df.head(args.limit)

    print(f"[INFO] Found {len(df)} files to process.")

    builder = DatasetBuilder()
    output_path.mkdir(parents=True, exist_ok=True)

    # Required columns check
    required_cols = ["midi_filename"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] CSV missing required column: {col}")
            sys.exit(1)

    success_count = 0
    fail_count = 0
    skip_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        midi_rel_path = row["midi_filename"]
        full_midi_path = data_root / midi_rel_path
        
        target_file = output_path / f"{full_midi_path.stem}.tokens.txt"

        if target_file.exists() and not args.overwrite:
            skip_count += 1
            continue

        if not full_midi_path.exists():
            fail_count += 1
            log_error(error_log_path, str(midi_rel_path), "File not found")
            continue

        try:
            # Extract metadata safely
            genre = str(row.get("genre", "UNKNOWN"))
            style = str(row.get("style", "UNKNOWN"))
            artist = str(row.get("artist", "UNKNOWN"))
            inst_type = str(row.get("inst_type", "UNKNOWN"))

            # Handle NaN/None
            if genre == "nan": genre = "UNKNOWN"
            if style == "nan": style = "UNKNOWN"
            if artist == "nan": artist = "UNKNOWN"
            if inst_type == "nan": inst_type = "UNKNOWN"

            builder.build(
                midi_path=str(full_midi_path),
                output_path=str(output_path), # Pass directory, Exporter handles filename
                genre=genre,
                style=style,
                artist=artist,
                inst_type=inst_type
            )
            
            if target_file.exists():
                success_count += 1
            else:
                fail_count += 1
                log_error(error_log_path, str(midi_rel_path), "Output file not created (Validation failed?)")

        except Exception as exc:
            # print(f"\n[ERROR] Failed {full_midi_path.name}: {exc}")
            fail_count += 1
            log_error(error_log_path, str(midi_rel_path), str(exc))

    print(f"\n[DONE] Success: {success_count}, Skipped: {skip_count}, Failed: {fail_count}")
    print(f"[INFO] Errors logged to: {error_log_path}")


if __name__ == "__main__":
    main()
