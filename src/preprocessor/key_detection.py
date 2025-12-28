from __future__ import annotations

import math
from typing import Dict, Tuple

_NOTE_NAME_TO_SEMITONE: Dict[str, int] = {
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

_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def detect_key(note_sequence) -> str | None:
    notes = [note for note in getattr(note_sequence, "notes", []) if not note.is_drum]
    if not notes:
        return None

    pitch_class_energy = [0.0] * 12
    for note in notes:
        dur = max(0.0, float(note.end_time) - float(note.start_time))
        if dur <= 0:
            dur = 0.05
        pitch_class_energy[int(note.pitch) % 12] += dur

    if all(val == 0 for val in pitch_class_energy):
        return None

    best_score = -1e9
    best_key = 0
    best_mode = "MAJOR"

    for mode, profile in [("MAJOR", _MAJOR_PROFILE), ("MINOR", _MINOR_PROFILE)]:
        for key in range(12):
            rotated = profile[-key:] + profile[:-key]
            score = _correlation(pitch_class_energy, rotated)
            if score > best_score:
                best_score = score
                best_key = key
                best_mode = mode

    return f"{_SEMITONE_TO_NAME_FLAT[best_key]}_{best_mode}"


def update_key_signature(note_sequence, key_name: str) -> None:
    semitone, mode = parse_key_name(key_name)
    if semitone is None:
        return
    del note_sequence.key_signatures[:]
    ks = note_sequence.key_signatures.add()
    ks.time = 0.0
    ks.key = semitone
    ks.mode = 0 if mode == "MAJOR" else 1


def parse_key_name(key_name: str) -> Tuple[int | None, str | None]:
    if not key_name or "_" not in key_name:
        return None, None
    root, mode = key_name.split("_", 1)
    root = root.strip().replace("-", "b")
    mode = mode.strip().upper()
    if mode not in ("MAJOR", "MINOR"):
        return None, None
    semitone = _NOTE_NAME_TO_SEMITONE.get(root)
    return semitone, mode


def _correlation(a, b) -> float:
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    den_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if den_a == 0 or den_b == 0:
        return -1e9
    return num / (den_a * den_b)
