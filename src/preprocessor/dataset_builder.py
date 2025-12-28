from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from music21 import converter as m21converter

try:
    from .chord_progression import extract_chord_grid
    from .exporter import bundle_to_text
    from .instrument_roles import infer_instrument_roles
    from .midi_metadata import (
        DEFAULT_BPM,
        DEFAULT_KEY,
        DEFAULT_TIME_SIGNATURE,
        format_key,
        get_key_from_sequence,
        get_tempo_bpm,
        get_time_signature,
        infer_key,
    )
except ImportError:  # Allows running as a script without package context.
    from chord_progression import extract_chord_grid
    from exporter import bundle_to_text
    from instrument_roles import infer_instrument_roles
    from midi_metadata import (
        DEFAULT_BPM,
        DEFAULT_KEY,
        DEFAULT_TIME_SIGNATURE,
        format_key,
        get_key_from_sequence,
        get_tempo_bpm,
        get_time_signature,
        infer_key,
    )

log = logging.getLogger(__name__)

DEFAULT_GRID_UNIT = "1/16"
DEFAULT_CTRL = "DYN:MID DEN:NORMAL MOV:STATIC FEEL:STRAIGHT"


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

    key_obj = get_key_from_sequence(note_sequence)
    if key_obj is None:
        m21_stream = m21converter.parse(str(midi_path))
        key_obj = infer_key(m21_stream)
    key_str = format_key(key_obj)

    prog_grid = extract_chord_grid(
        note_sequence,
        key_obj,
        (ts_num, ts_den),
        grid_unit,
    )

    sections = _segment_prog_grid(
        prog_grid,
        bars_per_section,
        default_ctrl,
    )

    bundle = {
        "global": {
            "BPM": bpm or DEFAULT_BPM,
            "KEY": key_str or DEFAULT_KEY,
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
) -> List[Dict]:
    sections: List[Dict] = []
    if bars_per_section <= 0:
        bars_per_section = max(1, len(prog_grid))
    for idx in range(0, len(prog_grid), bars_per_section):
        label = _section_label(idx // bars_per_section)
        bars = prog_grid[idx : idx + bars_per_section]
        sections.append(
            {
                "name": label,
                "bars": len(bars),
                "prog_grid": bars,
                "ctrl": default_ctrl,
            }
        )
    return sections


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
