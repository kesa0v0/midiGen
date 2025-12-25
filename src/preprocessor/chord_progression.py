from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

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

    def __init__(
        self,
        grid_unit: str = "1/4",
        min_notes: int = 2,
        min_confidence: float = 0.2,
        bass_role_weight: float = 8.0,
        min_pitch_boost: float = 2.0,
        sus_penalty: float = 0.03,
        sus_third_penalty: float = 0.3,
        sus_tiebreak_margin: float = 0.1,
        diatonic_bonus: float = 0.12,
        nondiatonic_penalty: float = 0.08,
        melody_penalty: float = 0.2,
        harmony_boost: float = 3.0,
        bass_root_bonus: float = 0.1,
    ):
        self.grid_unit = grid_unit
        self.min_notes = min_notes
        self.min_confidence = min_confidence
        self.bass_role_weight = max(1.0, float(bass_role_weight))
        self.min_pitch_boost = max(0.0, float(min_pitch_boost))
        self.sus_penalty = max(0.0, float(sus_penalty))
        self.sus_third_penalty = max(0.0, float(sus_third_penalty))
        self.sus_tiebreak_margin = max(0.0, float(sus_tiebreak_margin))
        self.diatonic_bonus = max(0.0, float(diatonic_bonus))
        self.nondiatonic_penalty = max(0.0, float(nondiatonic_penalty))
        self.melody_penalty = melody_penalty
        self.harmony_boost = harmony_boost
        self.bass_root_bonus = bass_root_bonus

        # 1. Detection Templates (Weighted intervals for accuracy)
        self.templates = {
            "maj": {0: 1.0, 4: 1.0, 7: 0.5},
            "min": {0: 1.0, 3: 1.0, 7: 0.5},
            "dim": {0: 1.0, 3: 1.0, 6: 0.5},
            "aug": {0: 1.0, 4: 1.0, 8: 0.5},      # Detect aug
            "sus2": {0: 1.0, 2: 0.8, 7: 0.5},     # Detect sus2
            "sus4": {0: 1.0, 5: 0.8, 7: 0.5},
            "7": {0: 1.0, 4: 1.0, 7: 0.5, 10: 0.8},
            "maj7": {0: 1.0, 4: 1.0, 7: 0.5, 11: 0.8},
            "m7": {0: 1.0, 3: 1.0, 7: 0.5, 10: 0.8},
            "dim7": {0: 1.0, 3: 1.0, 6: 0.5, 9: 0.8},   # Detect dim7
            "m7b5": {0: 1.0, 3: 1.0, 6: 0.5, 10: 0.8},  # Detect m7b5
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

        role_tracks = analysis.get("role_tracks") or {}
        bass_track_ids = set(role_tracks.get("BASS") or [])
        melody_track_ids = set(role_tracks.get("MELODY") or [])
        harmony_track_ids = set(role_tracks.get("HARMONY") or [])

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
                chord = self._chord_for_window(
                    midi,
                    s_time,
                    e_time,
                    min_notes=self.min_notes,
                    bass_track_ids=bass_track_ids,
                    melody_track_ids=melody_track_ids,
                    harmony_track_ids=harmony_track_ids,
                    key=roman_key,
                )
                
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
    def _chord_for_window(
        self,
        midi: pretty_midi.PrettyMIDI,
        start: float,
        end: float,
        min_notes: int = 1,
        bass_track_ids: Optional[Set[int]] = None,
        melody_track_ids: Optional[Set[int]] = None,
        harmony_track_ids: Optional[Set[int]] = None,
        key: Optional[str] = None,
    ) -> Optional[ChordCandidate]:
        hist = np.zeros(12, dtype=float)
        note_count = 0
        
        # Track lowest pitch as a fallback/bonus
        lowest_pitch_class = -1
        global_min_pitch = 128
        
        for inst_idx, inst in enumerate(midi.instruments):
            if inst.is_drum:
                continue
            
            weight_multiplier = 1.0
            if bass_track_ids and inst_idx in bass_track_ids:
                weight_multiplier = self.bass_role_weight
            elif melody_track_ids and inst_idx in melody_track_ids:
                weight_multiplier = self.melody_penalty
            elif harmony_track_ids and inst_idx in harmony_track_ids:
                weight_multiplier = self.harmony_boost

            for note in inst.notes:
                if note.start >= end or note.end <= start:
                    continue
                overlap = min(note.end, end) - max(note.start, start)
                if overlap <= 0:
                    continue
                weight = note.velocity * overlap * weight_multiplier
                pitch_class = note.pitch % 12
                hist[pitch_class] += weight
                note_count += 1
                
                # Update global lowest pitch
                if note.pitch < global_min_pitch:
                    global_min_pitch = note.pitch
                    lowest_pitch_class = pitch_class

        if hist.sum() == 0 or note_count < min_notes:
            return None
            
        # Pre-compute diatonic set if key is available
        diatonic_pc = set()
        if key and key != "UNKNOWN":
            tonic_pc, mode = self._parse_key(key)
            # Major Scale Intervals
            intervals = {0, 2, 4, 5, 7, 9, 11}
            if mode == "MINOR":
                # Natural Minor + Raised 7th (Harmonic) + Raised 6th (Melodic)
                intervals = {0, 2, 3, 5, 7, 8, 9, 10, 11}
            diatonic_pc = {(tonic_pc + i) % 12 for i in intervals}

        best: Optional[ChordCandidate] = None
        best_simple: Optional[ChordCandidate] = None
        for root in range(12):
            for quality, intervals in self.templates.items():
                tpl = np.zeros(12)
                for interval, weight in intervals.items():
                    tpl[(root + interval) % 12] = weight
                score = self._cosine_similarity(hist, tpl)
                
                # Diatonic Bonus / Non-Diatonic Penalty
                if diatonic_pc:
                    # Check if root is diatonic
                    if root in diatonic_pc:
                        score += self.diatonic_bonus
                    else:
                        score -= self.nondiatonic_penalty

                # Bass Root Bonus
                if lowest_pitch_class != -1 and root == lowest_pitch_class:
                    score += self.bass_root_bonus

                # Minimal triad bias; sus handled via penalties below.
                if quality in ["maj", "min"]:
                    score += 0.02
                if quality in ["sus2", "sus4"]:
                    score -= self.sus_penalty
                    third_strength = max(
                        hist[(root + 3) % 12],
                        hist[(root + 4) % 12],
                    )
                    if hist.sum() > 0 and third_strength >= 0.2 * hist.sum():
                        score -= self.sus_third_penalty
                # 7th, sus4 등은 점수가 확실히 높을 때만 선택됨
                if quality in ["maj", "min"]:
                    if best_simple is None or score > best_simple.score:
                        best_simple = ChordCandidate(root, quality, score)
                if best is None or score > best.score:
                    best = ChordCandidate(root, quality, score)
        if (
            best
            and best.quality in ["sus2", "sus4"]
            and best_simple
            and (best.score - best_simple.score) <= self.sus_tiebreak_margin
        ):
            best = ChordCandidate(best_simple.root_pc, best_simple.quality, best_simple.score)
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
