#!/usr/bin/env python
"""
Preprocess a single MIDI file into conductor tokens.

Usage:
    python scripts/preprocess_single.py path/to/file.mid --output data/processed/tokens
"""
import argparse
import sys
from pathlib import Path

# Make preprocessor modules importable
ROOT = Path(__file__).resolve().parents[1]
PREPROC_PATH = ROOT / "src" / "preprocessor"
if str(PREPROC_PATH) not in sys.path:
    sys.path.insert(0, str(PREPROC_PATH))

from dataset_builder import DatasetBuilder  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess a single MIDI file into tokens")
    parser.add_argument(
        "input_path",
        help="Path to the input MIDI file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/processed/tokens",
        help="Output directory (default: data/processed/tokens)",
    )
    parser.add_argument(
        "--genre",
        default="UNKNOWN",
        help="Genre metadata",
    )
    parser.add_argument(
        "--style",
        default="UNKNOWN",
        help="Style metadata",
    )
    parser.add_argument(
        "--artist",
        default="UNKNOWN",
        help="Artist metadata",
    )
    parser.add_argument(
        "--inst-type",
        default="UNKNOWN",
        help="Instrument type metadata",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    print(f"[INFO] Processing {input_path}...")
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    builder = DatasetBuilder()
    
    try:
        builder.build(
            midi_path=str(input_path),
            output_path=str(output_path),
            genre=args.genre,
            style=args.style,
            artist=args.artist,
            inst_type=args.inst_type
        )
        print(f"[DONE] Processing complete. Check output in {output_path}")
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
