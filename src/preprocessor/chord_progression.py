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
    Strategic Compression Chord Extraction.
    - Detection: High resolution (includes aug, dim7, etc.)
    - Output: Compressed to 8 core tokens (Strategic Vocabulary)
    - Mapping: Strict Chromatic Scale Degree (No functional interpretation)
    - Resolution: Beat-level (1 chord per beat) to prevent hyperactivity
    """

    def _slots_per_bar(self, bar: Dict) -> int:
        num, den = bar.get("time_sig", (4, 4))
        return int(num)

    def __init__(self, grid_unit: str = "1/4", min_notes: int = 2, min_confidence: float = 0.2):
        self.grid_unit = grid_unit
        self.min_notes = min_notes
        self.min_confidence = min_confidence

        # 1. Detection Templates (Broad detection for accuracy)
        self.templates = {
            "maj": [0, 4, 7],
            "min": [0, 3, 7],
            "dim": [0, 3, 6],
            "aug": [0, 4, 8],          # Detect aug
            "sus2": [0, 2, 7],         # Detect sus2
            "sus4": [0, 5, 7],
            "7": [0, 4, 7, 10],
            "maj7": [0, 4, 7, 11],
            "m7": [0, 3, 7, 10],
            "dim7": [0, 3, 6, 9],      # Detect dim7
            "m7b5": [0, 3, 6, 10],     # Detect m7b5
        }

        # 2. Strategic Vocabulary Mapping (Compression)
        # Raw Quality -> Target Quality (Llama Vocabulary)
        self.quality_map = {
            # Major Family -> I
            "maj": "maj",
            "aug": "maj",  # Aug -> Major (Color removed)
            "sus2": "maj", # Sus2 -> Major (Color removed)
            
            # Minor Family -> i
            "min": "min",
            
            # 7th Family -> I7 / i7 / Imaj7
            "7": "7",
            "m7": "m7",
            "maj7": "maj7",
            
            # Sus4 -> Isus4 (Function preserved)
            "sus4": "sus4",
            
            # Diminished Family -> io7
            "dim": "dim7", # Triad dim -> dim7 (Standardization)
            "dim7": "dim7",
            
            # Half-diminished -> ih7 (Function preserved)
            "m7b5": "m7b5",
        }

    def extract(self, midi: pretty_midi.PrettyMIDI, section: Section, analysis: Dict, slots_per_bar: int) -> List[List[str]]:
        bars = analysis.get("bars", [])
        if not bars:
            return []

        section_bars = bars[section.start_bar : section.end_bar]
        global_key = analysis.get("global_key")
        # Roman 변환 시에는 global_key만 사용 (section_key가 'KEEP'이거나 None이면 global_key)
        roman_key = section.local_key or global_key
        absolute_chords = bool(analysis.get("absolute_chords", False))
        
        # Ensure slots_per_bar is valid
        slots_per_bar = max(1, int(slots_per_bar))

        running_chord: Optional[ChordCandidate] = None
        prog_grid: List[List[str]] = []

        for bar in section_bars:
            bar_tokens: List[str] = []
            
            # Calculate beats to ensure beat-level resolution
            num, den = bar.get("time_sig", (4, 4))
            beats = max(1, int(num))
            
            # Calculate slots per beat (e.g., 4 slots / 4 beats = 1 slot/beat)
            slots_per_beat = max(1, slots_per_bar // beats)
            beat_dur = (bar["end"] - bar["start"]) / beats

            for b in range(beats):
                s_time = bar["start"] + b * beat_dur
                e_time = s_time + beat_dur

                # Detect chord for this beat
                chord = self._chord_for_window(midi, s_time, e_time, min_notes=self.min_notes)
                
                # Stabilization: Use running chord if detection is weak/empty
                if chord is None or chord.score < self.min_confidence:
                    chord = running_chord
                else:
                    running_chord = chord

                # Tokenize: N.C. or Roman (항상 global_key 기준)
                if chord is None:
                    token = "N.C."
                elif absolute_chords:
                    target_quality = self.quality_map.get(chord.quality, "maj")
                    token = self._absolute_chord(chord.root_pc, target_quality)
                else:
                    token = self._roman(chord, roman_key)
                
                # Fill slots for this beat
                # First slot gets the token, others get sustain "-"
                bar_tokens.append(token)
                for _ in range(slots_per_beat - 1):
                    bar_tokens.append("-")

            # Handle remainder slots if slots_per_bar isn't perfectly divisible
            if len(bar_tokens) < slots_per_bar:
                bar_tokens.extend(["-"] * (slots_per_bar - len(bar_tokens)))
            elif len(bar_tokens) > slots_per_bar:
                bar_tokens = bar_tokens[:slots_per_bar]

            prog_grid.append(bar_tokens)

        return prog_grid

    # ---- Chord detection ----
    def _chord_for_window(self, midi: pretty_midi.PrettyMIDI, start: float, end: float, min_notes: int = 1) -> Optional[ChordCandidate]:
        hist = np.zeros(12, dtype=float)
        note_count = 0
        
        min_pitch = 128
        bass_weight = 0.0

        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                if note.start >= end or note.end <= start:
                    continue
                overlap = min(note.end, end) - max(note.start, start)
                if overlap <= 0:
                    continue
                weight = note.velocity * overlap
                hist[note.pitch % 12] += weight
                note_count += 1
                
                # Bass Boost Logic: Track the lowest pitch in the window
                if note.pitch < min_pitch:
                    min_pitch = note.pitch
                    bass_weight = weight
                elif note.pitch == min_pitch:
                    # If multiple notes have the same lowest pitch, take the max weight
                    bass_weight = max(bass_weight, weight)

        if hist.sum() == 0 or note_count < min_notes:
            return None
            
        # Apply Bass Boost: Add extra weight to the bass note's pitch class
        # Adding 2.0 * bass_weight brings the total contribution of the bass to ~3x (1x original + 2x boost)
        if min_pitch < 128:
            hist[min_pitch % 12] += bass_weight * 2.0

        best: Optional[ChordCandidate] = None
        for root in range(12):
            for quality, intervals in self.templates.items():
                tpl = np.zeros(12)
                tpl[[ (root + i) % 12 for i in intervals ]] = 1.0
                score = self._cosine_similarity(hist, tpl)
                # Triad Bias: 단순 코드(maj, min)에 가산점, sus4는 소폭 가산점
                if quality in ["maj", "min"]:
                    score += 0.15
                elif quality in ["sus4"]:
                    score += 0.05
                # 7th, sus4 등은 점수가 확실히 높을 때만 선택됨
                if best is None or score > best.score:
                    best = ChordCandidate(root, quality, score)
        return best

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ---- Roman numeral Conversion (Strict) ----
    def _roman(self, chord: ChordCandidate, key: Optional[str]) -> str:
        root_pc = chord.root_pc
        
        # 1. Simplify Quality (Map to 8 core types)
        target_quality = self.quality_map.get(chord.quality, "maj")
        
        if not key or key == "UNKNOWN":
            return self._absolute_chord(root_pc, target_quality)

        tonic_pc, mode = self._parse_key(key)
        degree = (root_pc - tonic_pc) % 12

        # 2. Get Base Roman (I, bII, etc.) with Casing
        numeral = self._degree_to_roman(degree, target_quality)
        
        # 3. Apply Suffixes for Allowed Extensions
        if target_quality == "maj7":
            numeral += "maj7"
        elif target_quality == "7":
            numeral += "7"
        elif target_quality == "m7":
            numeral += "7"      # i + 7 = i7
        elif target_quality == "sus4":
            numeral += "sus4"
        elif target_quality == "dim7":
            numeral += "o7"     # viio7
        elif target_quality == "m7b5":
            numeral += "h7"     # iih7 (or iø7)

        return numeral

    def _degree_to_roman(self, degree: int, quality: str) -> str:
        # Fixed Chromatic Mapping (0-11)
        base_degrees = {
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
        
        roman = base_degrees.get(degree % 12, "?")

        # Casing Rule: Minor/Diminished families -> Lowercase
        if quality in ["min", "dim", "dim7", "m7", "m7b5"]:
            roman = roman.lower()
            
        return roman

    def _absolute_chord(self, root_pc: int, quality: str) -> str:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        name = names[root_pc % 12]
        
        suffix_map = {
            "maj": "", "min": "m", 
            "maj7": "maj7", "7": "7", "m7": "m7",
            "sus4": "sus4", 
            "dim7": "dim7", "m7b5": "m7b5"
        }
        suffix = suffix_map.get(quality, "")
        return name + suffix

    def _parse_key(self, key: str) -> Tuple[int, str]:
        tonic, mode = key.split("_")
        names = {
            "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
            "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
            "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
        }
        return names.get(tonic.upper(), 0), mode.upper()
