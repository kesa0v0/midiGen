import argparse
import json
from pathlib import Path
from typing import Optional


DEFAULT_INSTRUCTION = (
    "Generate a structured symbolic music representation based on the following metadata."
)


def extract_global_metadata(text: str) -> dict:
    in_global = False
    metadata = {}

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[GLOBAL]":
            in_global = True
            continue
        if in_global:
            if stripped.startswith("[") and stripped.endswith("]"):
                break
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key.strip().upper()] = value.strip()

    return metadata


def humanize_value(value: str) -> str:
    if not value:
        return value
    return value.replace("_", " ").title()


def build_input_sentence(metadata: dict, genre_fallback: Optional[str]) -> Optional[str]:
    title = metadata.get("TITLE")
    artist = metadata.get("ARTIST")
    key = metadata.get("KEY")
    bpm = metadata.get("BPM")
    genre = metadata.get("GENRE") or genre_fallback

    if not all([title, artist, key, bpm]):
        return None

    parts = []
    if genre:
        parts.append(f"Genre: {humanize_value(genre)}")
    parts.append(f"Key: {humanize_value(key)}")
    parts.append(f"Title: {title}")
    parts.append(f"Artist: {artist}")
    parts.append(f"BPM: {bpm}")

    return ", ".join(parts)


def iter_txt_files(dataset_dir: Path):
    for path in sorted(dataset_dir.rglob("*.txt")):
        if path.is_file():
            yield path


def convert_dataset(
    dataset_dir: Path,
    output_path: Path,
    instruction: str,
    require_genre: bool,
    genre_fallback: Optional[str],
) -> None:
    total_files = 0
    written = 0
    skipped_empty = 0
    skipped_missing = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for path in iter_txt_files(dataset_dir):
            total_files += 1
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                skipped_empty += 1
                continue

            if not text.strip():
                skipped_empty += 1
                continue

            metadata = extract_global_metadata(text)
            if require_genre and not metadata.get("GENRE"):
                skipped_missing += 1
                continue

            input_sentence = build_input_sentence(
                metadata, genre_fallback if not require_genre else None
            )
            if not input_sentence:
                skipped_missing += 1
                continue

            record = {
                "instruction": instruction,
                "input": input_sentence,
                "output": text,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(
        "Done. Total files:",
        total_files,
        "Written:",
        written,
        "Skipped empty:",
        skipped_empty,
        "Skipped missing metadata:",
        skipped_missing,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert conductor token .txt files to Alpaca-style JSONL."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/conductor_tokens"),
        help="Root directory containing .txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("train.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Instruction field for Alpaca-style records.",
    )
    parser.add_argument(
        "--require-genre",
        action="store_true",
        help="Skip files missing GENRE metadata.",
    )
    parser.add_argument(
        "--genre-fallback",
        type=str,
        default="Unknown",
        help="Fallback value when GENRE is missing (ignored with --require-genre).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    convert_dataset(
        dataset_dir=dataset_dir,
        output_path=args.output,
        instruction=args.instruction,
        require_genre=args.require_genre,
        genre_fallback=args.genre_fallback,
    )


if __name__ == "__main__":
    main()
