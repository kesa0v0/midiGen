from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import List, Tuple

try:
    from .key_detection import parse_key_name
except ImportError:  # Script execution fallback.
    from key_detection import parse_key_name

_NOTE_NAME_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
_SEMITONE_TO_NAME_FLAT = [
    "C",
    "Db",
    "D",
    "Eb",
    "E",
    "F",
    "Gb",
    "G",
    "Ab",
    "A",
    "Bb",
    "B",
]


def parse_fraction(text: str) -> Fraction:
    num, den = text.strip().split("/")
    return Fraction(int(num), int(den))


def steps_per_quarter(grid_unit: str) -> int:
    grid_fraction = parse_fraction(grid_unit)
    if grid_fraction <= 0:
        return 1
    quarter = Fraction(1, 4)
    steps = quarter / grid_fraction
    if steps.denominator != 1:
        steps = Fraction(int(round(float(steps))), 1)
    return max(1, int(steps))


def grid_steps_per_bar(time_sig: Tuple[int, int], grid_unit: str) -> int:
    bar_quarters = Fraction(time_sig[0] * 4, time_sig[1])
    steps = bar_quarters * steps_per_quarter(grid_unit)
    if steps.denominator != 1:
        steps = Fraction(int(round(float(steps))), 1)
    return max(1, int(steps))


def extract_chord_grid(
    note_sequence,
    key_name: str,
    time_sig: Tuple[int, int],
    grid_unit: str,
) -> List[List[str]]:
    try:
        from note_seq import chord_inference, sequences_lib
        from note_seq.protobuf import music_pb2
    except ImportError as exc:
        raise ImportError("note_seq is required for chord inference") from exc

    spq = steps_per_quarter(grid_unit)
    steps_per_bar = grid_steps_per_bar(time_sig, grid_unit)

    if not sequences_lib.is_quantized_sequence(note_sequence):
        note_sequence = sequences_lib.quantize_note_sequence(note_sequence, steps_per_quarter=spq)

    inferred = chord_inference.infer_chords_for_sequence(
        note_sequence,
        chords_per_bar=2,
        key_change_prob=0.001,
        chord_change_prob=0.5,
    )
    if inferred is None:
        inferred = note_sequence

    chord_events = []
    for ta in inferred.text_annotations:
        if ta.annotation_type == music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL:
            chord_events.append((float(ta.time), ta.text))
    chord_events.sort(key=lambda item: item[0])

    total_steps = _estimate_total_steps(note_sequence, spq, steps_per_bar)
    total_bars = max(1, int(math.ceil(total_steps / steps_per_bar)))
    total_steps = total_bars * steps_per_bar

    chord_by_step = _spread_chords(chord_events, total_steps, spq, note_sequence)
    prog_grid: List[List[str]] = []

    for bar_idx in range(total_bars):
        bar_tokens: List[str] = []
        prev_token = None
        bar_start_step = bar_idx * steps_per_bar
        for step in range(steps_per_bar):
            chord_symbol = chord_by_step[bar_start_step + step]
            token = chord_symbol_to_absolute(chord_symbol, key_name)
            if token == prev_token and token != "N.C.":
                bar_tokens.append("-")
            else:
                bar_tokens.append(token)
                prev_token = token
        prog_grid.append(bar_tokens)

    return prog_grid


def _estimate_total_steps(note_sequence, steps_per_quarter: int, steps_per_bar: int) -> int:
    if getattr(note_sequence, "notes", None):
        end_steps = [getattr(n, "quantized_end_step", None) for n in note_sequence.notes]
        end_steps = [s for s in end_steps if s is not None]
        if end_steps:
            return max(end_steps)
    total_time = float(getattr(note_sequence, "total_time", 0.0))
    qpm = _get_qpm(note_sequence)
    steps_per_second = (qpm / 60.0) * steps_per_quarter
    return max(steps_per_bar, int(round(total_time * steps_per_second)))


def _spread_chords(chord_events, total_steps: int, steps_per_quarter: int, note_sequence):
    chord_by_step = [None] * total_steps
    if not chord_events:
        return chord_by_step

    qpm = _get_qpm(note_sequence)
    steps_per_second = (qpm / 60.0) * steps_per_quarter

    events = []
    for time, symbol in chord_events:
        step = int(round(time * steps_per_second))
        step = max(0, min(step, total_steps - 1))
        events.append((step, symbol))
    events.sort(key=lambda item: item[0])

    current_symbol = None
    event_idx = 0
    for step in range(total_steps):
        while event_idx < len(events) and events[event_idx][0] <= step:
            current_symbol = events[event_idx][1]
            event_idx += 1
        chord_by_step[step] = current_symbol

    return chord_by_step


def _get_qpm(note_sequence) -> float:
    tempos = getattr(note_sequence, "tempos", [])
    if tempos:
        tempo = min(tempos, key=lambda t: t.time)
        return float(tempo.qpm)
    return 120.0


def chord_symbol_to_absolute(chord_symbol: str | None, key_name: str) -> str:
    if not chord_symbol or chord_symbol == "N.C.":
        return "N.C."

    root, quality, has_seventh = _parse_chord_symbol(chord_symbol)
    if root is None:
        return chord_symbol

    root_semitone = _NOTE_NAME_TO_SEMITONE.get(root)
    if root_semitone is None:
        return chord_symbol

    key_semitone, _ = parse_key_name(key_name)
    if key_semitone is None:
        key_semitone = 0

    transposed = (root_semitone - key_semitone) % 12
    root_name = _SEMITONE_TO_NAME_FLAT[transposed]
    return _format_chord_token(root_name, quality, has_seventh)


def _parse_chord_symbol(symbol: str):
    symbol = symbol.strip()
    if not symbol or symbol == "N.C.":
        return None, None, False
    root_part = symbol
    if ":" in symbol:
        root_part, qual = symbol.split(":", 1)
    else:
        root_part, qual = symbol, ""
    if "/" in root_part:
        root_part = root_part.split("/", 1)[0]
    if "/" in qual:
        qual = qual.split("/", 1)[0]

    match = re.match(r"^([A-G](?:#|b)?)(.*)$", root_part.strip())
    if match:
        root = match.group(1)
        remainder = match.group(2)
        qual = f"{remainder}{qual}"
    else:
        match = re.match(r"^([A-G](?:#|b)?)(.*)$", symbol)
        if not match:
            return None, None, False
        root = match.group(1)
        qual = match.group(2)

    qual_lower = qual.lower()
    has_seventh = "7" in qual_lower

    if "dim" in qual_lower or "o" in qual_lower:
        quality = "dim"
    elif "aug" in qual_lower or "+" in qual:
        quality = "aug"
    elif "min" in qual_lower or (qual_lower.startswith("m") and not qual_lower.startswith("maj")):
        quality = "min"
    else:
        quality = "maj"

    return root, quality, has_seventh


def _format_chord_token(root_name: str, quality: str | None, has_seventh: bool) -> str:
    suffix = ""
    if quality == "min":
        suffix = "m"
    elif quality == "dim":
        suffix = "dim"
    elif quality == "aug":
        suffix = "aug"
    if has_seventh:
        suffix = f"{suffix}7" if suffix else "7"
    return f"{root_name}{suffix}"
