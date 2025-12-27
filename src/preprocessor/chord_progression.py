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


@dataclass
class SlotInfo:
    bar_idx: int
    slot_idx: int
    candidates: List[ChordCandidate]
    bass_pc: Optional[int]
    bass_strength: float
    note_count: int


class ChordProgressionExtractor:

    """
    Strategic Compression Chord Extraction with Viterbi / HMM.
    - Detection: High resolution (includes aug, dim7, etc.)
    - Output: Compressed to 8 core tokens (Strategic Vocabulary)
    - Mapping: Strict Chromatic Scale Degree (No functional interpretation)
    - Resolution: Slot-level with hold tokens for grid-stable counts
    - Smoothing: Viterbi algorithm for context-aware chord selection
    - Root Anchor: Statistical reliance on strongest BASS track note for root detection
    - Inversion Aware: Supports recognition of 3rd/5th in bass
    """

    def _slots_per_bar(self, bar: Dict) -> int:
        num, den = bar.get("time_sig", (4, 4))
        return int(num)

    def __init__(
        self,
        grid_unit: str = "1/4",
        min_notes: int = 2,
        min_confidence: float = 0.2,
        bass_role_weight: float = 4.0,
        min_pitch_boost: float = 2.0,
        sus_penalty: float = 0.03,
        sus_third_penalty: float = 0.3,
        sus_tiebreak_margin: float = 0.1,
        diatonic_bonus: float = 0.12,
        nondiatonic_penalty: float = 0.08,
        diatonic_chord_penalty: float = 0.04,
        min_diatonic_chord_ratio: float = 0.5,
        melody_penalty: float = 0,
        harmony_boost: float = 2.5,
        bass_root_bonus: float = 2.0,
        triad_bonus: float = 0.05,
        same_chord_bonus: float = 0.25,
        change_penalty: float = 0.05,
        bass_hold_bonus: float = 0.3,
        bass_change_bonus: float = 0.05,
        use_role_tracks_only: bool = True,
        ignore_melody: bool = True,
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
        self.diatonic_chord_penalty = max(0.0, float(diatonic_chord_penalty))
        self.min_diatonic_chord_ratio = max(0.0, float(min_diatonic_chord_ratio))
        self.melody_penalty = melody_penalty
        self.harmony_boost = harmony_boost
        self.bass_root_bonus = bass_root_bonus
        self.triad_bonus = triad_bonus
        self.same_chord_bonus = max(0.0, float(same_chord_bonus))
        self.change_penalty = max(0.0, float(change_penalty))
        self.bass_hold_bonus = max(0.0, float(bass_hold_bonus))
        self.bass_change_bonus = max(0.0, float(bass_change_bonus))
        self.use_role_tracks_only = bool(use_role_tracks_only)
        self.ignore_melody = bool(ignore_melody)

        # 1. Detection Templates (Balanced for general musicality)
        self.templates = {
            # Triads: Base priority (1.0), allow 6th degree (0.5) for Pop/Jazz friendliness
            "maj": {0: 1.0, 4: 1.0, 7: 1.0, 9: 0.5},
            "min": {0: 1.0, 3: 1.0, 7: 1.0, 9: 0.5},

            # Dominant 7: High priority with guide tone emphasis
            "7": {0: 0.9, 4: 1.0, 7: 0.5, 10: 1.0},

            # 7th chords: Penalty relaxed (0.9)
            "maj7": {0: 0.9, 4: 0.9, 7: 0.5, 11: 0.9},
            "m7":   {0: 0.9, 3: 0.9, 7: 0.5, 10: 0.9},

            # Dim/Aug: Penalty reasonable (0.75 - 0.8)
            "dim":  {0: 0.75, 3: 0.75, 6: 0.75},
            "dim7": {0: 0.75, 3: 0.75, 6: 0.75, 9: 0.75},
            "m7b5": {0: 0.85, 3: 0.85, 6: 0.85, 10: 0.85},
            "aug":  {0: 0.8, 4: 0.8, 8: 0.8},

            # Sustain
            "sus4": {0: 0.9, 5: 0.9, 7: 0.5},
            "sus2": {0: 0.9, 2: 0.9, 7: 0.5},
        }

        # 2. Strategic Vocabulary Mapping (Compression)
        self.quality_map = {
            "maj": "maj", "aug": "maj", "sus2": "sus2",
            "min": "min",
            "7": "7", "m7": "m7", "maj7": "maj7",
            "sus4": "sus4",
            "dim": "dim7", "dim7": "dim7",
            "m7b5": "m7b5",
        }

    def extract(self, midi: pretty_midi.PrettyMIDI, section: Section, analysis: Dict, slots_per_bar: int) -> List[List[str]]:
        bars = analysis.get("bars", [])
        if not bars:
            return []

        section_bars = bars[section.start_bar : section.end_bar]
        global_key = analysis.get("global_key")
        roman_key = section.local_key or global_key
        absolute_chords = bool(analysis.get("absolute_chords", False))
        
        slots_per_bar = max(1, int(slots_per_bar))

        role_tracks = analysis.get("role_tracks") or {}
        bass_track_ids = set(role_tracks.get("BASS") or [])
        melody_track_ids = set(role_tracks.get("MELODY") or [])
        harmony_track_ids = set(role_tracks.get("HARMONY") or [])

        # --- Step 1: Collect candidates for all slots ---
        all_slots: List[SlotInfo] = []
        
        for bar_idx, bar in enumerate(section_bars):
            bar_dur = bar["end"] - bar["start"]
            slot_dur = bar_dur / slots_per_bar if slots_per_bar > 0 else 0.0

            for s in range(slots_per_bar):
                s_time = bar["start"] + s * slot_dur
                e_time = s_time + slot_dur

                candidates, bass_pc, bass_strength, note_count = self._chords_for_window(
                    midi,
                    s_time,
                    e_time,
                    min_notes=self.min_notes,
                    bass_track_ids=bass_track_ids,
                    melody_track_ids=melody_track_ids,
                    harmony_track_ids=harmony_track_ids,
                    key=roman_key,
                    top_k=4
                )
                
                if not candidates:
                    candidates = [ChordCandidate(root_pc=0, quality="N.C.", score=0.0)]
                
                all_slots.append(
                    SlotInfo(
                        bar_idx=bar_idx,
                        slot_idx=s,
                        candidates=candidates,
                        bass_pc=bass_pc,
                        bass_strength=bass_strength,
                        note_count=note_count,
                    )
                )

        if not all_slots:
            return []

        # --- Step 2: Viterbi Algorithm ---
        T = len(all_slots)
        dp = [np.zeros(len(slot.candidates)) for slot in all_slots]
        backpointer = [[0] * len(slot.candidates) for slot in all_slots]
        
        for i, cand in enumerate(all_slots[0].candidates):
            dp[0][i] = cand.score

        for t in range(1, T):
            prev_slot = all_slots[t - 1]
            curr_slot = all_slots[t]
            prev_cands = prev_slot.candidates
            curr_cands = curr_slot.candidates
            
            for j, curr_cand in enumerate(curr_cands):
                max_score = -float('inf')
                best_prev_idx = 0
                
                for i, prev_cand in enumerate(prev_cands):
                    transition = self._get_transition_score(prev_cand, curr_cand, prev_slot, curr_slot)
                    score = dp[t-1][i] + transition + curr_cand.score
                    
                    if score > max_score:
                        max_score = score
                        best_prev_idx = i
                
                dp[t][j] = max_score
                backpointer[t][j] = best_prev_idx

        best_path_indices = [0] * T
        best_end_idx = int(np.argmax(dp[T-1]))
        best_path_indices[T-1] = best_end_idx
        
        for t in range(T-1, 0, -1):
            best_path_indices[t-1] = backpointer[t][best_path_indices[t]]

        # --- Step 3: Format Output (event + hold) ---
        slot_tokens: List[str] = []
        for idx in range(T):
            cand_idx = best_path_indices[idx]
            chord = all_slots[idx].candidates[cand_idx]

            if chord.quality == "N.C.":
                token = "N.C."
            elif absolute_chords:
                target_quality = self.quality_map.get(chord.quality, "maj")
                token = self._absolute_chord(chord.root_pc, target_quality)
            else:
                token = self._roman(chord, roman_key)
            slot_tokens.append(token)

        slot_tokens = self._merge_single_slot_changes(slot_tokens)
        slot_tokens = self._replace_short_nc(slot_tokens, max_len=1)

        prog_grid: List[List[str]] = []
        slot_idx = 0
        for _ in section_bars:
            bar_tokens: List[str] = []
            prev_token = None
            for s in range(slots_per_bar):
                if slot_idx >= len(slot_tokens):
                    token = "N.C."
                else:
                    token = slot_tokens[slot_idx]
                slot_idx += 1

                if s == 0:
                    bar_tokens.append(token)
                    prev_token = token
                    continue

                if token == prev_token:
                    bar_tokens.append("-")
                else:
                    bar_tokens.append(token)
                    prev_token = token

            prog_grid.append(bar_tokens)

        return prog_grid

    def _get_transition_score(
        self,
        prev: ChordCandidate,
        curr: ChordCandidate,
        prev_slot: SlotInfo,
        curr_slot: SlotInfo,
    ) -> float:
        if prev.quality == "N.C." or curr.quality == "N.C.":
            return 0.0
            
        interval = (curr.root_pc - prev.root_pc) % 12
        score = 0.0
        
        if interval == 5: score += 0.2
        elif interval == 7: score += 0.1
        elif interval in [1, 2, 10, 11]: score += 0.05
        elif interval == 0: score += 0.15
        else: score -= 0.05
            
        if "dim" in prev.quality and "maj" in curr.quality:
            if interval != 1: score -= 0.1

        same_chord = (prev.root_pc == curr.root_pc and prev.quality == curr.quality)
        if same_chord:
            score += self.same_chord_bonus
        else:
            score -= self.change_penalty

        bass_hold = (
            prev_slot.bass_pc is not None
            and curr_slot.bass_pc is not None
            and prev_slot.bass_pc == curr_slot.bass_pc
        )
        bass_change = (
            prev_slot.bass_pc is not None
            and curr_slot.bass_pc is not None
            and prev_slot.bass_pc != curr_slot.bass_pc
        )
        if bass_hold:
            score += self.bass_hold_bonus if same_chord else -self.bass_hold_bonus
        elif bass_change:
            score += self.bass_change_bonus if not same_chord else -self.bass_change_bonus

        return score

    def _merge_single_slot_changes(self, tokens: List[str]) -> List[str]:
        if len(tokens) < 3:
            return tokens
        merged = list(tokens)
        for i in range(1, len(merged) - 1):
            if merged[i] != merged[i - 1] and merged[i - 1] == merged[i + 1]:
                merged[i] = merged[i - 1]
        return merged

    def _replace_short_nc(self, tokens: List[str], max_len: int = 1) -> List[str]:
        if not tokens:
            return tokens
        replaced = list(tokens)
        i = 0
        while i < len(replaced):
            if replaced[i] != "N.C.":
                i += 1
                continue
            j = i + 1
            while j < len(replaced) and replaced[j] == "N.C.":
                j += 1
            run_len = j - i
            if run_len <= max_len:
                prev_tok = replaced[i - 1] if i > 0 else None
                next_tok = replaced[j] if j < len(replaced) else None
                fill = prev_tok if prev_tok and prev_tok != "N.C." else next_tok
                if fill and fill != "N.C.":
                    for k in range(i, j):
                        replaced[k] = fill
            i = j
        return replaced

    # ---- Chord detection ----
    def _chords_for_window(
        self,
        midi: pretty_midi.PrettyMIDI,
        start: float,
        end: float,
        min_notes: int = 1,
        bass_track_ids: Optional[Set[int]] = None,
        melody_track_ids: Optional[Set[int]] = None,
        harmony_track_ids: Optional[Set[int]] = None,
        key: Optional[str] = None,
        top_k: int = 4
    ) -> Tuple[List[ChordCandidate], Optional[int], float, int]:
        
        hist = np.zeros(12, dtype=float)
        bass_hist = np.zeros(12, dtype=float)
        has_bass = False
        note_count = 0
        bass_strength = 0.0
        use_role_only = self.use_role_tracks_only and (
            (bass_track_ids and len(bass_track_ids) > 0)
            or (harmony_track_ids and len(harmony_track_ids) > 0)
        )
        
        for inst_idx, inst in enumerate(midi.instruments):
            if inst.is_drum:
                continue
            
            weight_multiplier = 1.0
            is_bass_track = False
            
            if bass_track_ids and inst_idx in bass_track_ids:
                weight_multiplier = self.bass_role_weight
                is_bass_track = True
            elif harmony_track_ids and inst_idx in harmony_track_ids:
                weight_multiplier = self.harmony_boost
            elif melody_track_ids and inst_idx in melody_track_ids:
                if self.ignore_melody:
                    continue
                weight_multiplier = self.melody_penalty
            elif use_role_only:
                continue

            if weight_multiplier <= 0:
                continue

            for note in inst.notes:
                if note.start >= end or note.end <= start:
                    continue
                overlap = min(note.end, end) - max(note.start, start)
                if overlap <= 0:
                    continue
                
                energy = note.velocity * overlap
                weight = energy * weight_multiplier
                pc = note.pitch % 12
                hist[pc] += weight
                note_count += 1
                
                if is_bass_track:
                    bass_hist[pc] += energy
                    has_bass = True

        bass_strength = float(bass_hist.sum())
        if hist.sum() == 0 or note_count < min_notes:
            return [], None, bass_strength, note_count
            
        diatonic_pc = set()
        if key and key != "UNKNOWN":
            tonic_pc, mode = self._parse_key(key)
            intervals = {0, 2, 4, 5, 7, 9, 11}
            if mode == "MINOR":
                intervals = {0, 2, 3, 5, 7, 8, 9, 10, 11}
            diatonic_pc = {(tonic_pc + i) % 12 for i in intervals}

        # Root anchor: Strongest bass note determines the target bass pitch class
        target_bass_pc = -1
        if has_bass and bass_hist.sum() > 0:
            target_bass_pc = int(np.argmax(bass_hist))
        elif hist.sum() > 0:
            target_bass_pc = int(np.argmax(hist))
        slot_bass_pc = int(np.argmax(bass_hist)) if has_bass and bass_hist.sum() > 0 else None

        candidates: List[ChordCandidate] = []
        
        for root in range(12):
            for quality, intervals in self.templates.items():
                tpl = np.zeros(12)
                for interval, weight in intervals.items():
                    tpl[(root + interval) % 12] = weight
                score = self._cosine_similarity(hist, tpl)
                
                if diatonic_pc:
                    if root in diatonic_pc:
                        score += self.diatonic_bonus
                    else:
                        score -= self.nondiatonic_penalty
                    chord_pcs = [(root + interval) % 12 for interval in intervals.keys()]
                    if chord_pcs:
                        nondiatonic = sum(1 for pc in chord_pcs if pc not in diatonic_pc)
                        if nondiatonic:
                            score -= self.diatonic_chord_penalty * nondiatonic
                            ratio = (len(chord_pcs) - nondiatonic) / len(chord_pcs)
                            if ratio < self.min_diatonic_chord_ratio:
                                score -= self.diatonic_chord_penalty

                # Smart Bass Authority
                if target_bass_pc != -1:
                    # Case A: Bass is Root (Strongest match)
                    if target_bass_pc == root:
                        score += self.bass_root_bonus
                    
                    # Case B: Bass is 3rd/5th (Inversion, partial bonus)
                    else:
                        # Major/Dominant 3rd
                        if (quality in ["maj", "7", "maj7"]) and target_bass_pc == (root + 4) % 12:
                            score += (self.bass_root_bonus * 0.25)
                        # Minor 3rd
                        elif (quality in ["min", "m7"]) and target_bass_pc == (root + 3) % 12:
                            score += (self.bass_root_bonus * 0.25)
                        # 5th
                        elif target_bass_pc == (root + 7) % 12:
                            score += (self.bass_root_bonus * 0.15)

                if quality in ["maj", "min"]:
                    score += 0.1 # General triad preference
                
                if quality in ["sus2", "sus4"]:
                    score -= self.sus_penalty
                    third_strength = max(hist[(root + 3) % 12], hist[(root + 4) % 12])
                    if hist.sum() > 0 and third_strength >= 0.2 * hist.sum():
                        score -= self.sus_third_penalty
                
                if score > 0.05:
                    candidates.append(ChordCandidate(root, quality, score))
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k], slot_bass_pc, bass_strength, note_count

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ---- Roman numeral Conversion (Strict) ----
    def _roman(self, chord: ChordCandidate, key: Optional[str]) -> str:
        root_pc = chord.root_pc
        target_quality = self.quality_map.get(chord.quality, "maj")
        
        if not key or key == "UNKNOWN":
            return self._absolute_chord(root_pc, target_quality)

        tonic_pc, mode = self._parse_key(key)
        degree = (root_pc - tonic_pc) % 12
        numeral = self._degree_to_roman(degree, target_quality)
        
        if target_quality == "maj7": numeral += "maj7"
        elif target_quality == "7": numeral += "7"
        elif target_quality == "m7": numeral += "7"
        elif target_quality == "sus2": numeral += "sus2"
        elif target_quality == "sus4": numeral += "sus4"
        elif target_quality == "dim7": numeral += "o7"
        elif target_quality == "m7b5": numeral += "h7"

        return numeral

    def _degree_to_roman(self, degree: int, quality: str) -> str:
        base_degrees = {
            0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV",
            6: "bV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII",
        }
        roman = base_degrees.get(degree % 12, "?")
        if quality in ["min", "dim", "dim7", "m7", "m7b5"]:
            roman = roman.lower()
        return roman

    def _absolute_chord(self, root_pc: int, quality: str) -> str:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        name = names[root_pc % 12]
        suffix_map = {
            "maj": "", "min": "m", "maj7": "maj7", "7": "7", "m7": "m7",
            "sus2": "sus2", "sus4": "sus4", "dim7": "dim7", "m7b5": "m7b5"
        }
        return name + suffix_map.get(quality, "")

    def _parse_key(self, key: str) -> Tuple[int, str]:
        tonic, mode = key.split("_")
        names = {
            "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
            "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
            "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
        }
        return names.get(tonic.upper(), 0), mode.upper()
