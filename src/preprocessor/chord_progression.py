from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Dict, List, Tuple

from music21 import chord as m21chord
from music21 import roman as m21roman


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
    key_obj,
    time_sig: Tuple[int, int],
    grid_unit: str,
) -> List[List[str]]:
    try:
        from note_seq import sequences_lib
    except ImportError as exc:
        raise ImportError("note_seq is required for chord extraction") from exc

    spq = steps_per_quarter(grid_unit)
    steps_per_bar = grid_steps_per_bar(time_sig, grid_unit)

    quantized = sequences_lib.quantize_note_sequence(note_sequence, spq)
    notes = [note for note in quantized.notes if not note.is_drum]
    if not notes:
        return [["N.C."] * steps_per_bar]

    max_end_step = max(int(note.quantized_end_step) for note in notes)
    total_bars = max(1, int(math.ceil(max_end_step / steps_per_bar)))
    total_steps = total_bars * steps_per_bar

    step_pitches = _collect_step_pitches(notes, total_steps)
    prog_grid: List[List[str]] = []

    for bar_idx in range(total_bars):
        bar_tokens: List[str] = []
        prev_token = None
        bar_start_step = bar_idx * steps_per_bar
        for step in range(steps_per_bar):
            step_idx = bar_start_step + step
            pitches = step_pitches[step_idx]
            token = "N.C."
            if pitches:
                token = pitches_to_roman(pitches, key_obj)
            if token == prev_token and token != "N.C.":
                bar_tokens.append("-")
            else:
                bar_tokens.append(token)
                prev_token = token
        prog_grid.append(bar_tokens)

    return prog_grid


def _collect_step_pitches(notes, total_steps: int) -> List[List[int]]:
    start_events: List[List[int]] = [[] for _ in range(total_steps + 1)]
    end_events: List[List[int]] = [[] for _ in range(total_steps + 1)]
    for note in notes:
        start = max(0, int(note.quantized_start_step))
        end = max(start, int(note.quantized_end_step))
        if end <= start:
            continue
        if start < total_steps:
            start_events[start].append(int(note.pitch))
        if end < total_steps:
            end_events[end].append(int(note.pitch))

    active: Dict[int, int] = {}
    step_pitches: List[List[int]] = []
    for step in range(total_steps):
        for pitch in start_events[step]:
            active[pitch] = active.get(pitch, 0) + 1
        for pitch in end_events[step]:
            count = active.get(pitch, 0) - 1
            if count <= 0:
                active.pop(pitch, None)
            else:
                active[pitch] = count
        step_pitches.append(list(active.keys()))

    return step_pitches


def pitches_to_roman(pitches: List[int], key_obj) -> str:
    try:
        chord_obj = m21chord.Chord(sorted(set(pitches)))
        rn = m21roman.romanNumeralFromChord(chord_obj, key_obj)
        figure = rn.figure.replace(" ", "")
        return simplify_roman(figure)
    except Exception:
        return "N.C."


def simplify_roman(figure: str) -> str:
    parts = figure.split("/")
    simplified = []
    for part in parts:
        digits = re.findall(r"\d+", part)
        keep_seventh = any("7" in d for d in digits)
        part = re.sub(r"\d+", "", part)
        if keep_seventh:
            part = f"{part}7"
        simplified.append(part)
    return "/".join(simplified)
