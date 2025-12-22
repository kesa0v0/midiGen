#!/usr/bin/env python
"""
Batch preprocess MIDI files into conductor tokens.

Usage:
    python script/preprocess.py -i path/to/midi_or_dir -o out_dir
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
    parser = argparse.ArgumentParser(description="Batch preprocess MIDI files into tokens")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="MIDI file or directory containing MIDI files",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file or directory (if dir, will write <stem>.tokens.txt inside)",
    )
    parser.add_argument(
        "--pattern",
        default="*.mid,*.midi",
        help="Comma-separated glob patterns for MIDI files when input is a directory (default: *.mid,*.midi)",
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


def collect_inputs(input_path: Path, patterns):
    if input_path.is_file():
        return [input_path]
    files = []
    for pat in patterns:
        files.extend(input_path.rglob(pat))
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return sorted(unique)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}")
        sys.exit(1)

    patterns = [p.strip() for p in args.pattern.split(",") if p.strip()]
    midi_files = collect_inputs(input_path, patterns)
    if args.limit:
        midi_files = midi_files[: args.limit]

    if not midi_files:
        print("[INFO] No MIDI files found.")
        return

    builder = DatasetBuilder()
    output_path.mkdir(parents=True, exist_ok=True) if output_path.suffix == "" or output_path.is_dir() else None

    for midi_file in midi_files:
        try:
            target = output_path
            if output_path.is_dir() or output_path.suffix == "":
                target = output_path / f"{midi_file.stem}.tokens.txt"
            if target.exists() and not args.overwrite:
                print(f"[SKIP] Exists: {target}")
                continue
            print(f"[RUN ] {midi_file} -> {target}")
            builder.build(str(midi_file), str(target))
            if target.exists():
                print(f"[OK  ] {target}")
            else:
                print(f"[FAIL] No output produced for {midi_file}")
        except Exception as exc:
            print(f"[ERROR] {midi_file}: {exc}")


if __name__ == "__main__":
    main()
