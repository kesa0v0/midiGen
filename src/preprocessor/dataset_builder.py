from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List

try:
    from .chord_progression import extract_chord_grid
    from .exporter import bundle_to_text, format_prog_grid
    from .instrument_roles import infer_instrument_roles
    from .key_detection import detect_key
    from .midi_metadata import (
        DEFAULT_BPM,
        DEFAULT_KEY,
        DEFAULT_TIME_SIGNATURE,
        get_key_from_sequence,
        get_tempo_bpm,
        get_time_signature,
    )
except ImportError:  # Allows running as a script without package context.
    from chord_progression import extract_chord_grid
    from exporter import bundle_to_text, format_prog_grid
    from instrument_roles import infer_instrument_roles
    from key_detection import detect_key
    from midi_metadata import (
        DEFAULT_BPM,
        DEFAULT_KEY,
        DEFAULT_TIME_SIGNATURE,
        get_key_from_sequence,
        get_tempo_bpm,
        get_time_signature,
    )

log = logging.getLogger(__name__)

DEFAULT_GRID_UNIT = "1/16"
DEFAULT_CTRL = "DEN:LOW REG:MID DYN:SOFT"


def build_conductor_bundle(
    midi_path: Path,
    grid_unit: str = DEFAULT_GRID_UNIT,
    bars_per_section: int = 8,
    default_ctrl: str = DEFAULT_CTRL,
    genre: str | None = None,
    style: str | None = None,
) -> Dict:
    midi_path = Path(midi_path)
    try:
        from note_seq import midi_io
    except ImportError as exc:
        raise ImportError("note_seq is required to build conductor bundles") from exc

    note_sequence = midi_io.midi_file_to_note_sequence(str(midi_path))

    bpm = get_tempo_bpm(note_sequence)
    ts_num, ts_den = get_time_signature(note_sequence)
    time_sig = f"{ts_num}/{ts_den}"

    detected_key = detect_key(note_sequence)
    key_for_transpose = detected_key or get_key_from_sequence(note_sequence) or DEFAULT_KEY

    prog_grid = extract_chord_grid(
        note_sequence,
        key_for_transpose,
        (ts_num, ts_den),
        grid_unit,
    )

    section_ctrls = _compute_section_ctrls(
        note_sequence,
        len(prog_grid),
        bars_per_section,
        bpm or DEFAULT_BPM,
        ts_num,
        ts_den,
    )
    sections = _segment_prog_grid(
        prog_grid,
        bars_per_section,
        default_ctrl,
        section_ctrls,
    )

    bundle = {
        "global": {
            "BPM": bpm or DEFAULT_BPM,
            "KEY": DEFAULT_KEY,
            "TIME_SIG": time_sig or f"{DEFAULT_TIME_SIGNATURE[0]}/{DEFAULT_TIME_SIGNATURE[1]}",
            "GRID_UNIT": grid_unit,
        },
        "instruments": infer_instrument_roles(note_sequence),
        "form": [(s["name"], s["bars"]) for s in sections],
        "sections": sections,
    }

    if genre:
        bundle["global"]["GENRE"] = genre
    if style:
        bundle["global"]["STYLE"] = style

    return bundle


def export_bundle_to_text(bundle: Dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(bundle_to_text(bundle), encoding="utf-8")


def process_midi_dir(
    input_dir: Path,
    output_dir: Path,
    grid_unit: str = DEFAULT_GRID_UNIT,
    bars_per_section: int = 8,
    default_ctrl: str = DEFAULT_CTRL,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    midi_paths = list(input_dir.rglob("*.mid")) + list(input_dir.rglob("*.midi"))

    for midi_path in midi_paths:
        rel = midi_path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".txt")
        try:
            bundle = build_conductor_bundle(
                midi_path,
                grid_unit=grid_unit,
                bars_per_section=bars_per_section,
                default_ctrl=default_ctrl,
            )
            export_bundle_to_text(bundle, out_path)
        except Exception as exc:
            log.warning("Failed to process %s: %s", midi_path, exc)


def _segment_prog_grid(
    prog_grid: List[List[str]],
    bars_per_section: int,
    default_ctrl: str,
    section_ctrls: List[str] | None = None,
) -> List[Dict]:
    sections: List[Dict] = []
    section_bars: List[List[List[str]]] = []
    if bars_per_section <= 0:
        bars_per_section = max(1, len(prog_grid))
    for idx in range(0, len(prog_grid), bars_per_section):
        bars = prog_grid[idx : idx + bars_per_section]
        section_idx = idx // bars_per_section
        ctrl = default_ctrl
        if section_ctrls and section_idx < len(section_ctrls):
            ctrl = section_ctrls[section_idx]
        section_bars.append(bars)
        sections.append(
            {
                "name": "",
                "bars": len(bars),
                "prog_grid": bars,
                "ctrl": ctrl,
            }
        )
    labels = _generate_structure_labels(section_bars)
    for section, label in zip(sections, labels):
        section["name"] = label
    return sections


def _compute_section_ctrls(
    note_sequence,
    total_bars: int,
    bars_per_section: int,
    bpm: float,
    ts_num: int,
    ts_den: int,
) -> List[str]:
    if bars_per_section <= 0:
        bars_per_section = max(1, total_bars)
    seconds_per_bar = _seconds_per_bar(bpm, ts_num, ts_den)
    total_time = float(getattr(note_sequence, "total_time", 0.0))
    ctrls: List[str] = []
    for start_bar in range(0, total_bars, bars_per_section):
        bars = min(bars_per_section, total_bars - start_bar)
        start_time = start_bar * seconds_per_bar
        end_time = start_time + (bars * seconds_per_bar)
        if total_time > 0:
            end_time = min(end_time, total_time)
        ctrls.append(_get_section_control_features(note_sequence.notes, start_time, end_time))
    return ctrls


def _seconds_per_bar(bpm: float, ts_num: int, ts_den: int) -> float:
    if bpm <= 0:
        bpm = DEFAULT_BPM
    beats_per_bar = ts_num * (4.0 / max(ts_den, 1))
    return (60.0 / bpm) * beats_per_bar


def _get_section_control_features(midi_notes, start_time: float, end_time: float) -> str:
    if end_time <= start_time:
        return DEFAULT_CTRL
    section_notes = [n for n in midi_notes if start_time <= n.start_time < end_time]
    if not section_notes:
        return DEFAULT_CTRL

    duration_sec = end_time - start_time
    notes_per_sec = len(section_notes) / duration_sec
    if notes_per_sec < 2:
        den = "LOW"
    elif notes_per_sec < 6:
        den = "MID"
    else:
        den = "HIGH"

    avg_pitch = float(np.mean([n.pitch for n in section_notes]))
    if avg_pitch < 48:
        reg = "LOW"
    elif avg_pitch < 72:
        reg = "MID"
    else:
        reg = "HIGH"

    avg_vel = float(np.mean([n.velocity for n in section_notes]))
    if avg_vel < 60:
        dyn = "SOFT"
    elif avg_vel < 90:
        dyn = "MID"
    else:
        dyn = "LOUD"

    return f"DEN:{den} REG:{reg} DYN:{dyn}"


def _section_label(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return alphabet[index]
    label = ""
    while index >= 0:
        index, rem = divmod(index, len(alphabet))
        label = alphabet[rem] + label
        index -= 1
    return label


def _generate_structure_labels(section_bars: List[List[List[str]]]) -> List[str]:
    unique_progs: List[str] = []
    counts: List[int] = []
    labels: List[str] = []

    for bars in section_bars:
        prog_str = _progression_signature(bars)
        if prog_str in unique_progs:
            idx = unique_progs.index(prog_str)
            counts[idx] += 1
            base = _section_label(idx)
            label = f"{base}{counts[idx]}"
        else:
            unique_progs.append(prog_str)
            counts.append(1)
            base = _section_label(len(unique_progs) - 1)
            label = f"{base}1"
        labels.append(label)
    return labels


def _progression_signature(bars: List[List[str]]) -> str:
    return "\n".join(format_prog_grid(bars))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export MIDI to conductor tokens.")
    parser.add_argument("midi_path", type=Path, help="Path to a MIDI file")
    parser.add_argument("output_path", type=Path, help="Output .txt path")
    parser.add_argument("--grid-unit", default=DEFAULT_GRID_UNIT)
    parser.add_argument("--bars-per-section", type=int, default=8)
    parser.add_argument("--genre", default=None)
    parser.add_argument("--style", default=None)

    args = parser.parse_args()
    bundle = build_conductor_bundle(
        args.midi_path,
        grid_unit=args.grid_unit,
        bars_per_section=args.bars_per_section,
        genre=args.genre,
        style=args.style,
    )
    export_bundle_to_text(bundle, args.output_path)
