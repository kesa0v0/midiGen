from __future__ import annotations

import logging
from typing import Tuple

log = logging.getLogger(__name__)

DEFAULT_BPM = 120
DEFAULT_TIME_SIGNATURE = (4, 4)
DEFAULT_KEY = "C_MAJOR"

_SHARP_TO_FLAT = {
    "C#": "Db",
    "D#": "Eb",
    "F#": "Gb",
    "G#": "Ab",
    "A#": "Bb",
}

_KEY_INDEX_TO_NAME = [
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

_KEY_MODE_MAP = {
    0: "MAJOR",
    1: "MINOR",
}


def get_tempo_bpm(note_sequence) -> int:
    if getattr(note_sequence, "tempos", None):
        tempo = min(note_sequence.tempos, key=lambda t: t.time)
        return int(round(float(tempo.qpm)))
    return DEFAULT_BPM


def get_time_signature(note_sequence) -> Tuple[int, int]:
    if getattr(note_sequence, "time_signatures", None):
        ts = min(note_sequence.time_signatures, key=lambda t: t.time)
        return int(ts.numerator), int(ts.denominator)
    return DEFAULT_TIME_SIGNATURE


def get_key_from_sequence(note_sequence) -> str | None:
    if getattr(note_sequence, "key_signatures", None):
        ks = min(note_sequence.key_signatures, key=lambda k: k.time)
        tonic = _KEY_INDEX_TO_NAME[int(ks.key) % 12]
        mode = _KEY_MODE_MAP.get(int(ks.mode), "MAJOR")
        return f"{tonic}_{mode}"
    return None


def normalize_pitch_name(name: str) -> str:
    if name in _SHARP_TO_FLAT:
        return _SHARP_TO_FLAT[name]
    return name.replace("-", "b")
