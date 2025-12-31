import argparse
import csv
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pretty_midi
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def is_good_midi(
    midi_path: Path,
    min_duration: float,
    max_duration: float,
    min_instruments: int,
) -> bool:
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return False

    duration = pm.get_end_time()
    if duration < min_duration or duration > max_duration:
        return False

    time_sigs = pm.time_signature_changes
    if time_sigs:
        first = time_sigs[0]
        if first.numerator != 4 or first.denominator != 4:
            return False

    instruments = pm.instruments or []
    has_drum = any(inst.is_drum for inst in instruments)
    if not has_drum and len(instruments) < min_instruments:
        return False

    return True


def parse_composer_title(rel_path: Path) -> tuple[str, str]:
    parts = rel_path.parts
    if len(parts) >= 3:
        composer = parts[0].strip()
        title = parts[1].strip()
    elif len(parts) == 2:
        composer = parts[0].strip()
        title = Path(parts[1]).stem.strip()
    else:
        composer = "Unknown"
        title = rel_path.stem.strip()

    if not composer:
        composer = "Unknown"
    if not title:
        title = rel_path.stem.strip() or "Unknown"

    return composer, title


def iter_midi_files(source_root: Path):
    for root, _, files in os.walk(source_root):
        for name in files:
            if name.lower().endswith((".mid", ".midi")):
                yield Path(root) / name


def safe_copy(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)
        return dest

    stem = dest.stem
    suffix = dest.suffix
    for idx in range(1, 10000):
        candidate = dest.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate

    raise RuntimeError(f"Too many duplicates for {dest}")


def _evaluate_midi_path(payload):
    midi_path_str, min_duration, max_duration, min_instruments = payload
    midi_path = Path(midi_path_str)
    ok = is_good_midi(
        midi_path,
        min_duration=min_duration,
        max_duration=max_duration,
        min_instruments=min_instruments,
    )
    return midi_path_str, ok


def _iter_payloads(source_root, min_duration, max_duration, min_instruments):
    for midi_path in iter_midi_files(source_root):
        yield (str(midi_path), min_duration, max_duration, min_instruments)


def _resolve_worker_count(num_workers: int) -> int:
    if num_workers <= 0:
        return max(1, (os.cpu_count() or 2) - 1)
    return num_workers


def _make_progress(enabled: bool, total=None, desc="Scanning"):
    if not enabled or tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit="file")


def main():
    parser = argparse.ArgumentParser(
        description="Filter Lakh MIDI Clean and export good files + metadata CSV."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="Lakh_MIDI_Clean",
        help="Root folder for Lakh MIDI Clean.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Dataset_MVP_5000",
        help="Output folder for accepted files (used unless --no-copy).",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="data/processed/lakh_clean_manifest.csv",
        help="CSV path to write (midi_path, composer, title).",
    )
    parser.add_argument(
        "--list-output",
        type=str,
        default=None,
        help="Optional text file path to write accepted midi paths (one per line).",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=5000,
        help="Stop after collecting this many files.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=60.0,
        help="Minimum duration in seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=300.0,
        help="Maximum duration in seconds.",
    )
    parser.add_argument(
        "--min-instruments",
        type=int,
        default=3,
        help="Minimum number of instruments if no drum track exists.",
    )
    parser.add_argument(
        "--flat-copy",
        action="store_true",
        help="Copy accepted files into a flat target folder (may rename on collisions).",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy files, only produce CSV/list from the source paths.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Log progress every N accepted files (0 disables).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Worker processes for MIDI parsing (0=auto, 1=serial).",
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=0,
        help="Max in-flight tasks for multiprocessing (0=auto).",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bar even if installed.",
    )

    args = parser.parse_args()

    source_root = Path(args.source)
    if not source_root.exists():
        print(f"Error: source folder not found: {source_root}")
        return 1

    copy_enabled = not args.no_copy
    target_root = Path(args.target)
    if copy_enabled:
        target_root.mkdir(parents=True, exist_ok=True)

    csv_path_input = Path(args.csv_output)
    csv_is_dir = str(args.csv_output).endswith(("/", "\\"))
    if csv_path_input.exists() and csv_path_input.is_dir():
        csv_is_dir = True
    if csv_is_dir:
        csv_path = csv_path_input / "lakh_clean_manifest.csv"
    else:
        csv_path = csv_path_input
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    list_output_path = Path(args.list_output) if args.list_output else None
    if list_output_path:
        list_output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    list_lines = []
    kept = 0
    scanned = 0

    num_workers = _resolve_worker_count(args.num_workers)
    use_pool = num_workers > 1
    max_pending = args.max_pending
    if max_pending <= 0:
        max_pending = num_workers * 4 if use_pool else 0

    show_progress = not args.no_tqdm
    if show_progress and tqdm is None:
        print("tqdm is not installed; progress bar disabled.")
        show_progress = False

    progress = _make_progress(show_progress)
    try:
        if not use_pool:
            for midi_path in iter_midi_files(source_root):
                scanned += 1
                if progress is not None:
                    progress.update(1)
                if not is_good_midi(
                    midi_path,
                    min_duration=args.min_duration,
                    max_duration=args.max_duration,
                    min_instruments=args.min_instruments,
                ):
                    continue

                try:
                    rel_path = midi_path.relative_to(source_root)
                except ValueError:
                    rel_path = Path(midi_path.name)

                composer, title = parse_composer_title(rel_path)

                output_path = midi_path
                if copy_enabled:
                    if args.flat_copy:
                        dest_path = target_root / midi_path.name
                    else:
                        dest_path = target_root / rel_path
                    output_path = safe_copy(midi_path, dest_path)

                rows.append(
                    {
                        "midi_path": str(output_path),
                        "composer": composer,
                        "title": title,
                    }
                )
                if list_output_path:
                    list_lines.append(str(output_path))

                kept += 1
                if progress is not None and kept % 50 == 0:
                    progress.set_postfix(kept=kept)
                if args.progress_every and kept % args.progress_every == 0:
                    print(f"Accepted {kept} files so far... (scanned={scanned})")

                if kept >= args.max_count:
                    print("Max count reached, stopping.")
                    break
        else:
            payloads = _iter_payloads(
                source_root,
                args.min_duration,
                args.max_duration,
                args.min_instruments,
            )
            futures = set()

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                try:
                    while len(futures) < max_pending:
                        futures.add(executor.submit(_evaluate_midi_path, next(payloads)))
                except StopIteration:
                    pass

                while futures and kept < args.max_count:
                    for future in as_completed(futures):
                        futures.remove(future)
                        try:
                            midi_path_str, ok = future.result()
                        except Exception:
                            ok = False
                            midi_path_str = None

                        scanned += 1
                        if progress is not None:
                            progress.update(1)

                        if not ok or not midi_path_str:
                            pass
                        else:
                            midi_path = Path(midi_path_str)
                            try:
                                rel_path = midi_path.relative_to(source_root)
                            except ValueError:
                                rel_path = Path(midi_path.name)

                            composer, title = parse_composer_title(rel_path)

                            output_path = midi_path
                            if copy_enabled:
                                if args.flat_copy:
                                    dest_path = target_root / midi_path.name
                                else:
                                    dest_path = target_root / rel_path
                                output_path = safe_copy(midi_path, dest_path)

                            rows.append(
                                {
                                    "midi_path": str(output_path),
                                    "composer": composer,
                                    "title": title,
                                }
                            )
                            if list_output_path:
                                list_lines.append(str(output_path))

                            kept += 1
                            if progress is not None and kept % 50 == 0:
                                progress.set_postfix(kept=kept)
                            if args.progress_every and kept % args.progress_every == 0:
                                print(f"Accepted {kept} files so far... (scanned={scanned})")

                            if kept >= args.max_count:
                                print("Max count reached, stopping.")
                                break

                        try:
                            futures.add(executor.submit(_evaluate_midi_path, next(payloads)))
                        except StopIteration:
                            pass

                    if kept >= args.max_count:
                        break

                if kept >= args.max_count:
                    executor.shutdown(wait=False, cancel_futures=True)
    finally:
        if progress is not None:
            progress.close()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["midi_path", "composer", "title"])
        writer.writeheader()
        writer.writerows(rows)

    if list_output_path:
        with open(list_output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(list_lines))

    print(f"Done. Accepted {kept} files (scanned={scanned}).")
    print(f"CSV saved to: {csv_path}")
    if list_output_path:
        print(f"List saved to: {list_output_path}")
    if copy_enabled:
        print(f"Copied files to: {target_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
