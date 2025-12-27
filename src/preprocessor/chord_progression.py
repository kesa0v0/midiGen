from __future__ import annotations

import math
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
    margin: float


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
        sus_strict_min: float = 0.18,
        sus_strict_max_third: float = 0.08,
        sus_strict_penalty: float = 0.25,
        no3_third_max: float = 0.1,
        no3_fifth_min: float = 0.12,
        no3_bonus: float = 0.05,
        no3_penalty: float = 0.2,
        diatonic_bonus: float = 0.12,
        nondiatonic_penalty: float = 0.08,
        diatonic_chord_penalty: float = 0.04,
        min_diatonic_chord_ratio: float = 0.5,
        melody_penalty: float = 0.2,
        harmony_boost: float = 2.5,
        bass_root_bonus: float = 2.0,
        triad_bonus: float = 0.05,
        same_chord_bonus: float = 0.25,
        change_penalty: float = 0.05,
        bass_hold_bonus: float = 0.3,
        bass_change_bonus: float = 0.05,
        use_role_tracks_only: bool = True,
        ignore_melody: bool = True,
        downsample_min_duration_bars: float = 0.5,
        downsample_confidence_keep: float = 0.35,
        bass_consistency_confidence: float = 0.45,
        downbeat_bonus: float = 0.2,
    ):
        self.grid_unit = grid_unit
        self.min_notes = min_notes
        self.min_confidence = min_confidence
        self.bass_role_weight = max(1.0, float(bass_role_weight))
        self.min_pitch_boost = max(0.0, float(min_pitch_boost))
        self.sus_penalty = max(0.0, float(sus_penalty))
        self.sus_third_penalty = max(0.0, float(sus_third_penalty))
        self.sus_tiebreak_margin = max(0.0, float(sus_tiebreak_margin))
        self.sus_strict_min = max(0.0, float(sus_strict_min))
        self.sus_strict_max_third = max(0.0, float(sus_strict_max_third))
        self.sus_strict_penalty = max(0.0, float(sus_strict_penalty))
        self.no3_third_max = max(0.0, float(no3_third_max))
        self.no3_fifth_min = max(0.0, float(no3_fifth_min))
        self.no3_bonus = max(0.0, float(no3_bonus))
        self.no3_penalty = max(0.0, float(no3_penalty))
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
        self.downsample_min_duration_bars = max(0.0, float(downsample_min_duration_bars))
        self.downsample_confidence_keep = max(0.0, float(downsample_confidence_keep))
        self.bass_consistency_confidence = max(0.0, float(bass_consistency_confidence))
        self.downbeat_bonus = float(downbeat_bonus)

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

            # No 3rd (Power chord)
            "no3":  {0: 1.0, 7: 1.0},
        }

        # 2. Strategic Vocabulary Mapping (Compression)
        self.quality_map = {
            "maj": "maj", "aug": "maj", "sus2": "sus2",
            "min": "min",
            "7": "7", "m7": "m7", "maj7": "maj7",
            "sus4": "sus4",
            "dim": "dim7", "dim7": "dim7",
            "m7b5": "m7b5",
            "no3": "5",
        }

    def extract(
        self,
        midi: pretty_midi.PrettyMIDI,
        section: Section,
        analysis: Dict,
        detect_slots_per_bar: int,
        export_slots_per_bar: Optional[int] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[str]]], int]:
        bars = analysis.get("bars", [])
        if not bars:
            return [], None, max(1, int(export_slots_per_bar or detect_slots_per_bar or 1))

        section_bars = bars[section.start_bar : section.end_bar]
        global_key = analysis.get("global_key")
        roman_key = section.local_key or global_key
        absolute_chords = bool(analysis.get("absolute_chords", False))
        
        detect_slots_per_bar = max(1, int(detect_slots_per_bar))
        if export_slots_per_bar is None:
            export_slots_per_bar = detect_slots_per_bar
        export_slots_per_bar = max(1, int(export_slots_per_bar))
        if export_slots_per_bar > detect_slots_per_bar:
            export_slots_per_bar = detect_slots_per_bar

        role_tracks = analysis.get("role_tracks") or {}
        bass_track_ids = set(role_tracks.get("BASS") or [])
        melody_track_ids = set(role_tracks.get("MELODY") or [])
        harmony_track_ids = set(role_tracks.get("HARMONY") or [])

        # --- Step 1: Collect candidates for all slots ---
        all_slots: List[SlotInfo] = []
        
        for bar_idx, bar in enumerate(section_bars):
            bar_dur = bar["end"] - bar["start"]
            slot_dur = bar_dur / detect_slots_per_bar if detect_slots_per_bar > 0 else 0.0

            for s in range(detect_slots_per_bar):
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
                margin = self._candidate_margin(candidates)
                all_slots.append(
                    SlotInfo(
                        bar_idx=bar_idx,
                        slot_idx=s,
                        candidates=candidates,
                        bass_pc=bass_pc,
                        bass_strength=bass_strength,
                        note_count=note_count,
                        margin=margin,
                    )
                )

        if not all_slots:
            return [], None, export_slots_per_bar

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

        # --- Step 3: Downsample + Format Output ---
        chosen_chords: List[ChordCandidate] = []
        slot_confidence: List[float] = []
        slot_bass: List[Optional[int]] = []
        for idx in range(T):
            cand_idx = best_path_indices[idx]
            chord = all_slots[idx].candidates[cand_idx]
            chosen_chords.append(chord)
            slot_confidence.append(self._slot_confidence(all_slots[idx]))
            slot_bass.append(all_slots[idx].bass_pc)

        chords_out = chosen_chords
        conf_out = slot_confidence
        bass_out = slot_bass

        if export_slots_per_bar < detect_slots_per_bar:
            chords_out, conf_out, bass_out = self._downsample_chords(
                chosen_chords,
                slot_confidence,
                slot_bass,
                detect_slots_per_bar,
                export_slots_per_bar,
            )

        chords_out = self._apply_bass_consistency(chords_out, conf_out, bass_out)
        chords_out = self._apply_min_duration(chords_out, conf_out, export_slots_per_bar)
        chords_out = self._merge_single_slot_changes_chords(chords_out)
        chords_out = self._replace_short_nc_chords(chords_out, max_len=1)

        split_mode = analysis.get("chord_detail_mode") == "split"
        slot_tokens, ext_tokens = self._render_tokens(
            chords_out,
            roman_key,
            absolute_chords,
            split_mode,
        )

        prog_grid = self._format_grid(slot_tokens, export_slots_per_bar, section_bars)
        prog_ext_grid = None
        if ext_tokens is not None:
            prog_ext_grid = self._format_grid(ext_tokens, export_slots_per_bar, section_bars)

        return prog_grid, prog_ext_grid, export_slots_per_bar

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

    def _candidate_margin(self, candidates: List[ChordCandidate]) -> float:
        if not candidates:
            return 0.0
        if len(candidates) == 1:
            return float(candidates[0].score)
        return float(candidates[0].score - candidates[1].score)

    def _slot_confidence(self, slot: SlotInfo) -> float:
        return max(0.0, float(slot.margin))

    def _downsample_chords(
        self,
        chords: List[ChordCandidate],
        confidences: List[float],
        bass_pcs: List[Optional[int]],
        detect_slots_per_bar: int,
        export_slots_per_bar: int,
    ) -> Tuple[List[ChordCandidate], List[float], List[Optional[int]]]:
        if export_slots_per_bar <= 0:
            return chords, confidences, bass_pcs
        if detect_slots_per_bar % export_slots_per_bar != 0:
            return chords, confidences, bass_pcs
        group = detect_slots_per_bar // export_slots_per_bar
        if group <= 1:
            return chords, confidences, bass_pcs

        out_chords: List[ChordCandidate] = []
        out_conf: List[float] = []
        out_bass: List[Optional[int]] = []

        for i in range(0, len(chords), group):
            group_chords = chords[i : i + group]
            group_conf = confidences[i : i + group]
            group_bass = bass_pcs[i : i + group]
            chosen, conf, bass_pc = self._choose_group_chord(group_chords, group_conf, group_bass)
            out_chords.append(chosen)
            out_conf.append(conf)
            out_bass.append(bass_pc)

        return out_chords, out_conf, out_bass

    def _choose_group_chord(
        self,
        group_chords: List[ChordCandidate],
        group_conf: List[float],
        group_bass: List[Optional[int]],
    ) -> Tuple[ChordCandidate, float, Optional[int]]:
        if not group_chords:
            return ChordCandidate(root_pc=0, quality="N.C.", score=0.0), 0.0, None

        scores: Dict[Tuple[int, str], float] = {}
        representatives: Dict[Tuple[int, str], ChordCandidate] = {}
        total_weight = 0.0

        for idx, chord in enumerate(group_chords):
            weight = 1.0 + min(1.0, max(0.0, group_conf[idx] if idx < len(group_conf) else 0.0))
            if idx == 0:
                weight += self.downbeat_bonus
            if (
                idx < len(group_bass)
                and group_bass[idx] is not None
                and chord.quality != "N.C."
                and group_bass[idx] == chord.root_pc
            ):
                weight += 0.1
            key = (chord.root_pc, chord.quality)
            scores[key] = scores.get(key, 0.0) + weight
            representatives.setdefault(key, chord)
            total_weight += weight

        best_key = max(scores.items(), key=lambda item: item[1])[0]
        best_score = scores[best_key]
        confidence = best_score / total_weight if total_weight > 0 else 0.0
        bass_pc = self._mode_pc(group_bass)
        return representatives[best_key], confidence, bass_pc

    def _mode_pc(self, values: List[Optional[int]]) -> Optional[int]:
        counts: Dict[int, int] = {}
        for val in values:
            if val is None:
                continue
            counts[val] = counts.get(val, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda item: item[1])[0]

    def _apply_bass_consistency(
        self,
        chords: List[ChordCandidate],
        confidences: List[float],
        bass_pcs: List[Optional[int]],
    ) -> List[ChordCandidate]:
        if not chords:
            return chords
        adjusted = list(chords)
        for i in range(1, len(adjusted)):
            if i >= len(bass_pcs):
                break
            prev_bass = bass_pcs[i - 1]
            curr_bass = bass_pcs[i]
            if prev_bass is None or curr_bass is None:
                continue
            if prev_bass != curr_bass:
                continue
            if self._chord_equal(adjusted[i], adjusted[i - 1]):
                continue
            if i < len(confidences) and confidences[i] < self.bass_consistency_confidence:
                adjusted[i] = adjusted[i - 1]
        return adjusted

    def _apply_min_duration(
        self,
        chords: List[ChordCandidate],
        confidences: List[float],
        slots_per_bar: int,
    ) -> List[ChordCandidate]:
        if not chords:
            return chords
        min_slots = int(math.ceil(self.downsample_min_duration_bars * max(1, slots_per_bar)))
        min_slots = max(1, min_slots)

        adjusted = list(chords)
        i = 0
        while i < len(adjusted):
            j = i + 1
            while j < len(adjusted) and self._chord_equal(adjusted[j], adjusted[i]):
                j += 1
            run_len = j - i
            run_conf = max(confidences[i:j] or [0.0])
            if run_len < min_slots and run_conf < self.downsample_confidence_keep:
                prev_chord = adjusted[i - 1] if i > 0 else None
                next_chord = adjusted[j] if j < len(adjusted) else None
                fill = prev_chord or next_chord
                if fill is not None:
                    for k in range(i, j):
                        adjusted[k] = fill
            i = j
        return adjusted

    def _merge_single_slot_changes_chords(self, chords: List[ChordCandidate]) -> List[ChordCandidate]:
        if len(chords) < 3:
            return chords
        merged = list(chords)
        for i in range(1, len(merged) - 1):
            if not self._chord_equal(merged[i], merged[i - 1]) and self._chord_equal(merged[i - 1], merged[i + 1]):
                merged[i] = merged[i - 1]
        return merged

    def _replace_short_nc_chords(self, chords: List[ChordCandidate], max_len: int = 1) -> List[ChordCandidate]:
        if not chords:
            return chords
        replaced = list(chords)
        i = 0
        while i < len(replaced):
            if replaced[i].quality != "N.C.":
                i += 1
                continue
            j = i + 1
            while j < len(replaced) and replaced[j].quality == "N.C.":
                j += 1
            run_len = j - i
            if run_len <= max_len:
                prev_chord = replaced[i - 1] if i > 0 else None
                next_chord = replaced[j] if j < len(replaced) else None
                fill = prev_chord or next_chord
                if fill is not None and fill.quality != "N.C.":
                    for k in range(i, j):
                        replaced[k] = fill
            i = j
        return replaced

    def _render_tokens(
        self,
        chords: List[ChordCandidate],
        roman_key: Optional[str],
        absolute_chords: bool,
        split_mode: bool,
    ) -> Tuple[List[str], Optional[List[str]]]:
        slot_tokens: List[str] = []
        ext_tokens: Optional[List[str]] = [] if split_mode else None

        for chord in chords:
            if chord.quality == "N.C.":
                token = "N.C."
                ext_token = "N.C."
            elif split_mode:
                base_quality, ext = self._split_quality(chord.quality)
                if absolute_chords:
                    token = self._absolute_chord_base(chord.root_pc, base_quality)
                else:
                    token = self._roman_base(chord.root_pc, base_quality, roman_key)
                ext_token = ext or "NONE"
            elif absolute_chords:
                target_quality = self.quality_map.get(chord.quality, "maj")
                token = self._absolute_chord(chord.root_pc, target_quality)
                ext_token = None
            else:
                token = self._roman(chord, roman_key)
                ext_token = None

            slot_tokens.append(token)
            if ext_tokens is not None:
                ext_tokens.append(ext_token or "NONE")

        return slot_tokens, ext_tokens

    def _format_grid(
        self,
        slot_tokens: List[str],
        slots_per_bar: int,
        section_bars: List[Dict],
    ) -> List[List[str]]:
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

    def _split_quality(self, quality: str) -> Tuple[str, Optional[str]]:
        if quality in {"maj", "7", "maj7", "sus2", "sus4"}:
            return "maj", quality if quality not in {"maj"} else None
        if quality in {"min", "m7"}:
            return "min", "m7" if quality == "m7" else None
        if quality in {"dim", "dim7", "m7b5"}:
            ext = quality if quality != "dim" else None
            return "dim", ext
        if quality == "aug":
            return "aug", None
        if quality == "no3":
            return "5", "NO3"
        return "maj", None

    def _roman_base(self, root_pc: int, base_quality: str, key: Optional[str]) -> str:
        if not key or key == "UNKNOWN":
            return self._absolute_chord_base(root_pc, base_quality)

        tonic_pc, _ = self._parse_key(key)
        degree = (root_pc - tonic_pc) % 12
        numeral = self._degree_to_roman(degree, base_quality)
        if base_quality == "dim":
            numeral += "o"
        elif base_quality == "aug":
            numeral += "+"
        elif base_quality == "5":
            numeral += "5"
        return numeral

    def _absolute_chord_base(self, root_pc: int, quality: str) -> str:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        name = names[root_pc % 12]
        suffix_map = {
            "maj": "",
            "min": "m",
            "dim": "dim",
            "aug": "aug",
            "5": "5",
        }
        return name + suffix_map.get(quality, "")

    def _chord_equal(self, a: ChordCandidate, b: ChordCandidate) -> bool:
        return a.root_pc == b.root_pc and a.quality == b.quality

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
        hist_sum = float(hist.sum())
        if hist_sum == 0 or note_count < min_notes:
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
                third_strength = max(hist[(root + 3) % 12], hist[(root + 4) % 12])
                third_ratio = third_strength / hist_sum if hist_sum > 0 else 0.0
                fifth_ratio = hist[(root + 7) % 12] / hist_sum if hist_sum > 0 else 0.0
                
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
                    sus_pc = (root + 5) % 12 if quality == "sus4" else (root + 2) % 12
                    sus_ratio = hist[sus_pc] / hist_sum if hist_sum > 0 else 0.0
                    if third_ratio > self.sus_strict_max_third or sus_ratio < self.sus_strict_min:
                        score -= self.sus_strict_penalty
                    if third_ratio >= 0.2:
                        score -= self.sus_third_penalty
                elif quality == "no3":
                    if third_ratio > self.no3_third_max or fifth_ratio < self.no3_fifth_min:
                        score -= self.no3_penalty
                    else:
                        score += self.no3_bonus
                
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
        elif target_quality == "5": numeral += "5"

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
            "sus2": "sus2", "sus4": "sus4", "dim7": "dim7", "m7b5": "m7b5", "5": "5"
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
