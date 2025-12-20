from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi

from models import Section


@dataclass
class ChordCandidate:
    root_pc: int
    quality: str
    score: float


class ChordProgressionExtractor:
    """
    Heuristic chord extraction (bar-level, 1 chord per bar, non-ML).
    - Aggregate notes per bar
    - Build pitch class histogram
    - Pick best-fitting chord candidate
    - Apply section/local key (fallback to global) to Roman numeral
    - Fill bar grid with chord + sustains
    - If bar is unstable, reuse previous chord
    """

    def __init__(self, grid_unit: str = "1/4", min_notes: int = 2, min_confidence: float = 0.2):
        self.grid_unit = grid_unit
        self.min_notes = min_notes
        self.min_confidence = min_confidence

        # Chord templates (relative pitch-class sets)
        self.templates = {
            "maj": [0, 4, 7],
            "min": [0, 3, 7],
            "dim": [0, 3, 6],
            "sus2": [0, 2, 7],
            "sus4": [0, 5, 7],
            "7": [0, 4, 7, 10],
            "maj7": [0, 4, 7, 11],
            "m7": [0, 3, 7, 10],
        }

    def extract(self, midi: pretty_midi.PrettyMIDI, section: Section, analysis: Dict, slots_per_bar: int) -> List[List[str]]:
        bars = analysis.get("bars", [])
        if not bars:
            return []

        section_bars = bars[section.start_bar : section.end_bar]
        global_key = analysis.get("global_key")  # optional
        section_key = section.local_key or global_key
        grid_slots = max(1, int(slots_per_bar))

        prev_chord = None
        prog_grid: List[List[str]] = []
        for bar in section_bars:
            chord = self._chord_for_bar(midi, bar)
            if chord is None or chord.score < self.min_confidence:
                chord = prev_chord

            if chord is None:
                token = "N.C."
            else:
                token = self._roman(chord, section_key)
                prev_chord = chord

            prog_grid.append(self._fill_bar(token, grid_slots))

        return prog_grid

    # ---- Chord detection ----
    def _chord_for_bar(self, midi: pretty_midi.PrettyMIDI, bar: Dict) -> Optional[ChordCandidate]:
        start, end = bar["start"], bar["end"]
        hist = np.zeros(12, dtype=float)

        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                if start <= note.start < end:
                    overlap = max(0.0, min(note.end, end) - note.start)
                    weight = note.velocity * (overlap if overlap > 0 else 1.0)
                    hist[note.pitch % 12] += weight

        if hist.sum() == 0 or bar.get("note_count", 0) < self.min_notes:
            return None

        best: Optional[ChordCandidate] = None
        for root in range(12):
            for quality, intervals in self.templates.items():
                tpl = np.zeros(12)
                tpl[[ (root + i) % 12 for i in intervals ]] = 1.0
                score = self._cosine_similarity(hist, tpl)
                if best is None or score > best.score:
                    best = ChordCandidate(root, quality, score)
        return best

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ---- Roman numeral ----
    def _roman(self, chord: ChordCandidate, key: Optional[str]) -> str:
        root_pc = chord.root_pc
        quality = chord.quality
        if not key or key == "UNKNOWN":
            return self._absolute_chord(root_pc, quality)

        tonic_pc, mode = self._parse_key(key)
        degree = (root_pc - tonic_pc) % 12

        numeral = self._degree_to_roman(degree, mode, quality)
        if quality in ("7", "maj7", "m7"):
            suffix = "7" if quality == "7" else ("maj7" if quality == "maj7" else "7")
            numeral = numeral + suffix
        elif quality == "dim":
            numeral = numeral + "o"
        elif quality.startswith("sus"):
            numeral = numeral + quality

        return numeral

    def _absolute_chord(self, root_pc: int, quality: str) -> str:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        name = names[root_pc % 12]
        suffix = ""
        if quality == "min":
            suffix = "m"
        elif quality == "dim":
            suffix = "dim"
        elif quality == "7":
            suffix = "7"
        elif quality == "maj7":
            suffix = "maj7"
        elif quality == "m7":
            suffix = "m7"
        elif quality.startswith("sus"):
            suffix = quality
        return name + suffix

    def _parse_key(self, key: str) -> Tuple[int, str]:
        tonic, mode = key.split("_")
        names = {
            "C": 0,
            "C#": 1,
            "DB": 1,
            "D": 2,
            "D#": 3,
            "EB": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "GB": 6,
            "G": 7,
            "G#": 8,
            "AB": 8,
            "A": 9,
            "A#": 10,
            "BB": 10,
            "B": 11,
        }
        return names.get(tonic.upper(), 0), mode.upper()

    def _degree_to_roman(self, degree: int, mode: str, quality: str) -> str:
        base_major = {
            0: "I",
            1: "bII",
            2: "II",
            3: "bIII",
            4: "III",
            5: "IV",
            6: "bV",
            7: "V",
            8: "bVI",
            9: "VI",
            10: "bVII",
            11: "VII",
        }
        base = base_major[degree]
        if mode == "MINOR":
            diatonic_minor = {0: "i", 2: "ii", 3: "iii", 5: "iv", 7: "v", 8: "VI", 10: "VII"}
            if degree in diatonic_minor:
                base = diatonic_minor[degree]
        if quality == "min" and base.isupper():
            base = base.lower()
        return base

    # ---- Helpers ----
    def _fill_bar(self, chord: str, slots: int) -> List[str]:
        slots = max(1, slots)
        return [chord] + ["-"] * (slots - 1)

    def _key_from_bars(self, bars: List[Dict], global_key: Optional[str]) -> Optional[str]:
        # Deprecated: key is decided upstream (KeyDetector). Kept for compatibility.
        return global_key
