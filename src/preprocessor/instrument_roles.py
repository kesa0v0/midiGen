from __future__ import annotations

import logging
import re
from typing import Dict, List

import pretty_midi

log = logging.getLogger(__name__)


def infer_instrument_roles(note_sequence) -> Dict[str, str]:
    roles = {
        "MELODY": "UNKNOWN",
        "HARMONY": "UNKNOWN",
        "BASS": "UNKNOWN",
        "DRUMS": "NONE",
    }

    notes = getattr(note_sequence, "notes", [])
    has_drums = any(note.is_drum for note in notes)
    if has_drums:
        roles["DRUMS"] = "STANDARD_DRUMS"

    program_groups: Dict[int, List[int]] = {}
    for note in notes:
        if note.is_drum:
            continue
        program = int(getattr(note, "program", 0))
        program_groups.setdefault(program, []).append(int(note.pitch))

    if not program_groups:
        return roles

    stats = []
    for program, pitches in program_groups.items():
        avg_pitch = sum(pitches) / max(1, len(pitches))
        stats.append((program, avg_pitch, len(pitches)))

    bass_program = min(stats, key=lambda s: s[1])[0]
    roles["BASS"] = program_token(bass_program)

    remaining = [s for s in stats if s[0] != bass_program]
    if not remaining:
        roles["MELODY"] = roles["BASS"]
        roles["HARMONY"] = roles["BASS"]
        return roles

    melody_program = max(remaining, key=lambda s: s[1])[0]
    roles["MELODY"] = program_token(melody_program)

    remaining = [s for s in remaining if s[0] != melody_program]
    if not remaining:
        roles["HARMONY"] = roles["MELODY"]
        return roles

    harmony_program = max(remaining, key=lambda s: s[2])[0]
    roles["HARMONY"] = program_token(harmony_program)
    return roles


def program_token(program: int) -> str:
    try:
        name = pretty_midi.program_to_instrument_name(int(program))
    except Exception:
        name = f"PROGRAM_{int(program)}"
    token = normalize_instrument_name(name)
    if not token:
        token = "UNKNOWN"
    return token


def normalize_instrument_name(name: str) -> str:
    name = re.sub(r"[()]", " ", name)
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return name.strip("_").upper()
