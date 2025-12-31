import argparse
import csv
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


INVALID_PATH_CHARS = set('<>:"/\\|?*')


def _safe_name(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "Unknown"
    cleaned = []
    for ch in value:
        if ch in INVALID_PATH_CHARS or ord(ch) < 32:
            cleaned.append("_")
        else:
            cleaned.append(ch)
    return "".join(cleaned).strip() or "Unknown"


def _short_hash(text: str, length: int = 8) -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return digest[:length]


def _resolve_worker_count(num_workers: int) -> int:
    if num_workers <= 0:
        return max(1, (os.cpu_count() or 2) - 1)
    return num_workers


def _make_progress(enabled: bool, total=None, desc="Processing"):
    if not enabled or tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit="file")


def _count_manifest_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def _iter_manifest_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _output_path_from_row(
    midi_path: Path,
    output_dir: Path,
    source_root: Path | None,
    composer: str,
    title: str,
) -> Path:
    if source_root:
        try:
            rel = midi_path.relative_to(source_root)
        except ValueError:
            rel = None
        if rel is not None:
            return output_dir / rel.with_suffix(".txt")

    composer_dir = _safe_name(composer)
    title_dir = _safe_name(title)
    stem = _safe_name(midi_path.stem)
    suffix = _short_hash(str(midi_path))
    filename = f"{stem}__{suffix}.txt"
    return output_dir / composer_dir / title_dir / filename


def _process_row(payload):
    (
        midi_path_str,
        output_path_str,
        grid_unit,
        bars_per_section,
        default_ctrl,
        genre,
        style,
        title,
        composer,
    ) = payload

    midi_path = Path(midi_path_str)
    output_path = Path(output_path_str)

    try:
        from src.preprocessor.dataset_builder import build_conductor_bundle, export_bundle_to_text
    except ImportError:
        from preprocessor.dataset_builder import build_conductor_bundle, export_bundle_to_text

    bundle = build_conductor_bundle(
        midi_path,
        grid_unit=grid_unit,
        bars_per_section=bars_per_section,
        default_ctrl=default_ctrl,
        genre=genre,
        style=style,
    )
    bundle.setdefault("global", {})
    if title:
        bundle["global"]["TITLE"] = title
    if composer:
        bundle["global"]["ARTIST"] = composer
        bundle["global"]["COMPOSER"] = composer

    export_bundle_to_text(bundle, output_path)
    return True, None


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MIDI files listed in a manifest using dataset_builder."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest CSV (must include midi_path).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write exported text files.",
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default=None,
        help="If set, preserve relative paths from this root.",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default=None,
        help="Optional output CSV to log results.",
    )
    parser.add_argument(
        "--grid-unit",
        type=str,
        default="1/16",
        help="Grid unit for chord inference.",
    )
    parser.add_argument(
        "--bars-per-section",
        type=int,
        default=8,
        help="Bars per section for segmentation.",
    )
    parser.add_argument(
        "--default-ctrl",
        type=str,
        default="DEN:LOW REG:MID DYN:SOFT",
        help="Default control string for sections.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Worker processes (0=auto, 1=serial).",
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=0,
        help="Max in-flight tasks for multiprocessing (0=auto).",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip items whose output already exists.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Reprocess even if output exists.",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bar even if installed.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=0,
        help="Optional cap on the number of rows to process (0 = no limit).",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_root = Path(args.source_root) if args.source_root else None

    output_manifest = Path(args.output_manifest) if args.output_manifest else None
    if output_manifest:
        output_manifest.parent.mkdir(parents=True, exist_ok=True)

    show_progress = not args.no_tqdm
    if show_progress and tqdm is None:
        print("tqdm is not installed; progress bar disabled.")
        show_progress = False

    total_rows = _count_manifest_rows(manifest_path)
    progress = _make_progress(show_progress, total=total_rows)

    num_workers = _resolve_worker_count(args.num_workers)
    use_pool = num_workers > 1
    max_pending = args.max_pending
    if max_pending <= 0:
        max_pending = num_workers * 4 if use_pool else 0

    results = []
    processed = 0
    submitted = 0

    def record_result(
        midi_path: str,
        output_path: str,
        composer: str,
        title: str,
        status: str,
        error: str | None,
    ):
        results.append(
            {
                "midi_path": midi_path,
                "output_path": output_path,
                "composer": composer,
                "title": title,
                "status": status,
                "error": error or "",
            }
        )

    try:
        row_iter = _iter_manifest_rows(manifest_path)
        if not use_pool:
            for row in row_iter:
                midi_path_str = (row.get("midi_path") or "").strip()
                if not midi_path_str:
                    if progress is not None:
                        progress.update(1)
                    continue

                midi_path = Path(midi_path_str)
                composer = (row.get("composer") or row.get("artist") or "Unknown").strip()
                title = (row.get("title") or "Unknown").strip()
                genre = (row.get("genre") or "").strip() or None
                style = (row.get("style") or "").strip() or None

                output_path = _output_path_from_row(
                    midi_path,
                    output_dir,
                    source_root,
                    composer,
                    title,
                )

                if args.resume and output_path.exists():
                    record_result(midi_path_str, str(output_path), composer, title, "skipped", None)
                    if progress is not None:
                        progress.update(1)
                    continue

                ok = False
                error = None
                try:
                    ok, error = _process_row(
                        (
                            midi_path_str,
                            str(output_path),
                            args.grid_unit,
                            args.bars_per_section,
                            args.default_ctrl,
                            genre,
                            style,
                            title,
                            composer,
                        )
                    )
                except Exception as exc:
                    ok = False
                    error = str(exc)

                status = "ok" if ok else "error"
                record_result(midi_path_str, str(output_path), composer, title, status, error)

                processed += 1
                if progress is not None:
                    progress.update(1)
                    if processed % 50 == 0:
                        progress.set_postfix(done=processed)

                if args.max_count and processed >= args.max_count:
                    break
        else:
            futures = set()
            pending_rows = {}

            def submit_row(row):
                nonlocal submitted
                midi_path_str = (row.get("midi_path") or "").strip()
                if not midi_path_str:
                    record_result("", "", "", "", "missing", "missing midi_path")
                    if progress is not None:
                        progress.update(1)
                    return

                midi_path = Path(midi_path_str)
                composer = (row.get("composer") or row.get("artist") or "Unknown").strip()
                title = (row.get("title") or "Unknown").strip()
                genre = (row.get("genre") or "").strip() or None
                style = (row.get("style") or "").strip() or None

                output_path = _output_path_from_row(
                    midi_path,
                    output_dir,
                    source_root,
                    composer,
                    title,
                )

                if args.resume and output_path.exists():
                    record_result(midi_path_str, str(output_path), composer, title, "skipped", None)
                    if progress is not None:
                        progress.update(1)
                    return

                payload = (
                    midi_path_str,
                    str(output_path),
                    args.grid_unit,
                    args.bars_per_section,
                    args.default_ctrl,
                    genre,
                    style,
                    title,
                    composer,
                )
                future = executor.submit(_process_row, payload)
                futures.add(future)
                pending_rows[future] = (
                    midi_path_str,
                    str(output_path),
                    composer,
                    title,
                )
                submitted += 1

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                try:
                    while len(futures) < max_pending:
                        submit_row(next(row_iter))
                except StopIteration:
                    pass

                while futures:
                    for future in as_completed(futures):
                        futures.remove(future)

                        midi_path_str, output_path_str, composer, title = pending_rows.pop(
                            future, ("", "", "", "")
                        )

                        try:
                            ok, error = future.result()
                        except Exception as exc:
                            ok = False
                            error = str(exc)

                        status = "ok" if ok else "error"
                        record_result(midi_path_str, output_path_str, composer, title, status, error)

                        processed += 1
                        if progress is not None:
                            progress.update(1)
                            if processed % 50 == 0:
                                progress.set_postfix(done=processed)

                        if args.max_count and processed >= args.max_count:
                            futures.clear()
                            break

                        try:
                            submit_row(next(row_iter))
                        except StopIteration:
                            pass

                    if args.max_count and processed >= args.max_count:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
    finally:
        if progress is not None:
            progress.close()

    if output_manifest:
        with output_manifest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["midi_path", "output_path", "composer", "title", "status", "error"],
            )
            writer.writeheader()
            writer.writerows(results)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    skip_count = sum(1 for r in results if r["status"] == "skipped")
    err_count = sum(1 for r in results if r["status"] == "error")
    print(f"Done. ok={ok_count}, skipped={skip_count}, error={err_count}")
    if output_manifest:
        print(f"Output manifest saved to: {output_manifest}")
    print(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
