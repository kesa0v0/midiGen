from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import pretty_midi
from music21 import converter, stream, key as m21key, pitch as m21pitch


@dataclass(frozen=True)
class KeyResult:
    global_key: str  # e.g., "C_MAJOR" or "UNKNOWN"
    section_keys: Dict[str, str]  # section_id -> "KEEP" or "F#_MINOR"


class KeyStringNormalizer:
    """
    music21 Key 객체/문자열을 v3.0 KEY 토큰 형태로 정규화
    """
    @staticmethod
    def normalize(k: Optional[m21key.Key]) -> str:
        if k is None:
            return "UNKNOWN"

        tonic = k.tonic.name  # e.g. 'C', 'F#', 'B-'
        # music21 flats: 'B-' 형태가 흔함 → 'Bb'
        tonic = tonic.replace("-", "b")

        mode = k.mode  # 'major' / 'minor' / sometimes 'dorian' etc.
        if mode is None:
            return "UNKNOWN"

        mode_up = mode.upper()
        if mode_up not in ("MAJOR", "MINOR"):
            # 현재 단계에서는 modal key까지 억지로 넣지 말고 UNKNOWN 처리(오염 방지)
            return "UNKNOWN"

        return f"{tonic.upper()}_{mode_up}"


class KeyDetector:
    """
    Global key + per-section key candidate를 music21로 추정한다.
    결정은 보수적으로 하고, 불확실하면 UNKNOWN/KEEP로 남긴다.
    """
    def __init__(
        self,
        min_mod_bars: int = 4,
        stability_windows: int = 3,
        hist_override_margin: float = 0.08,
        low_pitch_threshold: int = 52,
        low_pitch_boost: float = 0.6,
        diatonic_override_margin: float = 0.08,
        diatonic_min_ratio: float = 0.55,
    ):
        self.min_mod_bars = min_mod_bars
        self.stability_windows = stability_windows
        self.hist_override_margin = hist_override_margin
        self.low_pitch_threshold = low_pitch_threshold
        self.low_pitch_boost = low_pitch_boost
        self.diatonic_override_margin = diatonic_override_margin
        self.diatonic_min_ratio = diatonic_min_ratio

    def detect(
        self,
        midi_path: str,
        sections: List[Tuple[str, int, int]],
        debug: bool = False,
        midi: Optional[pretty_midi.PrettyMIDI] = None,
        resolve_diatonic: bool = True,
    ) -> KeyResult:
        """
        sections: list of (section_id, start_bar_inclusive, end_bar_exclusive)
        bar index는 0-based, end_bar는 exclusive로 가정.
        """
        score = converter.parse(midi_path)
        
        # [Bass-Weighted Detection]
        # "민주주의의 함정" 해결: 베이스 파트의 비중을 높여 화성학적 뿌리를 강조
        score = self._apply_bass_weights(score, weight=5)

        global_key = self._detect_global_key(score, midi_path=midi_path, debug=debug)
        diatonic_override = False
        diatonic_key = None
        diatonic_ratio = None
        base_diatonic_ratio = None
        if resolve_diatonic and midi is not None:
            diatonic_hist = self._pc_histogram_pretty_midi(midi)
            if sum(diatonic_hist) > 0:
                tonic_pc, mode, ratio = self._best_key_by_diatonic_ratio(diatonic_hist)
                diatonic_ratio = ratio
                if tonic_pc is not None:
                    diatonic_key = m21key.Key(tonic=m21pitch.Pitch(tonic_pc), mode=mode)
                    if global_key is None:
                        diatonic_override = True
                        global_key = diatonic_key
                    else:
                        base_mode = global_key.mode if global_key.mode in ["major", "minor"] else "major"
                        base_diatonic_ratio = self._diatonic_ratio(
                            diatonic_hist,
                            global_key.tonic.pitchClass,
                            base_mode,
                        )
                        if (
                            diatonic_ratio - base_diatonic_ratio >= self.diatonic_override_margin
                            and diatonic_ratio >= self.diatonic_min_ratio
                        ):
                            diatonic_override = True
                            global_key = diatonic_key
        global_key_str = KeyStringNormalizer.normalize(global_key)
        if debug and resolve_diatonic and midi is not None:
            base_ratio_str = f"{base_diatonic_ratio:.3f}" if base_diatonic_ratio is not None else "NA"
            diatonic_ratio_str = f"{diatonic_ratio:.3f}" if diatonic_ratio is not None else "NA"
            diatonic_key_str = KeyStringNormalizer.normalize(diatonic_key)
            print(
                f"[KeyDetector] diatonic_key={diatonic_key_str} "
                f"diatonic_ratio={diatonic_ratio_str} base_ratio={base_ratio_str} "
                f"override_margin={self.diatonic_override_margin:.3f} "
                f"min_ratio={self.diatonic_min_ratio:.3f} "
                f"override={str(diatonic_override)}"
            )

        if diatonic_override:
            section_key_map = {sec_id: "KEEP" for sec_id, _, _ in sections}
            if debug:
                print("[KeyDetector] section_key_override=KEEP (diatonic override)")
            return KeyResult(global_key=global_key_str, section_keys=section_key_map)

        section_key_map: Dict[str, str] = {}
        for sec_id, start_bar, end_bar in sections:
            bars = max(0, end_bar - start_bar)
            if bars < self.min_mod_bars:
                # 너무 짧은 섹션은 전조로 확정하지 않고 KEEP
                section_key_map[sec_id] = "KEEP"
                continue

            candidate = self._detect_section_key(score, start_bar, end_bar)
            cand_str = KeyStringNormalizer.normalize(candidate)

            # 불확실하면 KEEP (데이터 오염 방지)
            if cand_str == "UNKNOWN" or global_key_str == "UNKNOWN":
                section_key_map[sec_id] = "KEEP"
                continue

            section_key_map[sec_id] = "KEEP" if cand_str == global_key_str else cand_str

        return KeyResult(global_key=global_key_str, section_keys=section_key_map)

    def _apply_bass_weights(self, score: stream.Score, weight: int = 5) -> stream.Score:
        """
        Identify bass parts and duplicate them to increase their influence on key detection.
        Combats the issue where high-frequency melody notes overwhelm the bass root.
        """
        bass_parts = []
        for p in score.parts:
            if self._is_bass_part(p):
                bass_parts.append(p)
        
        if not bass_parts:
            return score
            
        # Add (weight - 1) copies
        for bp in bass_parts:
            for _ in range(weight - 1):
                new_part = copy.deepcopy(bp)
                score.insert(0, new_part)
        
        return score

    def _is_bass_part(self, part: stream.Part) -> bool:
        # 1. Check Instrument Name / Program
        try:
            inst = part.getInstrument()
            if inst:
                # Check name
                inst_name = inst.bestName()
                if inst_name and "bass" in inst_name.lower():
                    return True
                # Check MIDI Program (32-39: Bass, 87: Bass & Lead)
                if inst.midiProgram is not None:
                    if 32 <= inst.midiProgram <= 39:
                        return True
        except Exception:
            pass
            
        # 2. Check Part Name
        if part.partName and "bass" in part.partName.lower():
            return True
            
        # 3. Check Average Pitch (Fallback)
        pitches = []
        try:
            # Flatten to find all notes
            for n in part.flatten().notes:
                if n.isNote:
                    pitches.append(n.pitch.midi)
                elif n.isChord:
                    for p in n.pitches:
                        pitches.append(p.midi)
        except Exception:
            pass
            
        if pitches:
            avg_pitch = sum(pitches) / len(pitches)
            if avg_pitch <= 48: # C3 (MIDI 48) or lower
                return True
                
        return False

    def _detect_global_key(
        self,
        score: stream.Score,
        midi_path: Optional[str] = None,
        debug: bool = False,
    ) -> Optional[m21key.Key]:
        # 1) music21 basic analysis
        m21_key = None
        try:
            k = score.analyze("key")
            if isinstance(k, m21key.Key):
                m21_key = k
        except Exception:
            pass
        
        if not m21_key:
             # Fallback to analyzing first 16 measures if full score analysis failed
            try:
                seg = score.measures(1, 16)
                k = seg.analyze("key")
                if isinstance(k, m21key.Key):
                    m21_key = k
            except Exception:
                pass

        # 2) Histogram fallback and override
        hist = self._pc_histogram(
            score,
            low_pitch_threshold=self.low_pitch_threshold,
            low_pitch_boost=self.low_pitch_boost,
        )
        raw_hist = self._pc_histogram(score) if debug else None
        hist_key = None
        hist_score = 0.0
        if sum(hist) > 0:
            tonic_pc, mode, hist_score = self._best_key_from_histogram(hist)
            if tonic_pc is not None:
                hist_key = m21key.Key(tonic=m21pitch.Pitch(tonic_pc), mode=mode)

        base_key = m21_key
        m21_score = None
        if hist_key is not None:
            if m21_key is None:
                base_key = hist_key
            else:
                m21_mode = m21_key.mode if m21_key.mode in ["major", "minor"] else "major"
                m21_score = self._key_profile_score(
                    hist,
                    m21_key.tonic.pitchClass,
                    m21_mode,
                )
                if hist_score - m21_score >= self.hist_override_margin:
                    base_key = hist_key

        # 3) Heuristic Correction: Start/End Bass Note
        # Pop music often starts and ends on the Tonic (I).
        # If the bass note of the first and last measure match, use it as the tonic.
        start_bass = None
        end_bass = None
        bass_override_key = None
        try:
            start_bass = self._get_measure_bass(score, 1)
            
            # Find last measure number.
            # Usually score.measures() gives access, but finding the index of the last measure is tricky.
            # We try to get it from the parts.
            last_measure_num = 0
            if score.parts:
                for p in score.parts:
                    # Note: This is an approximation. 
                    # Correctly finding the last measure with content can be complex.
                    try:
                        # getElementsByClass('Measure') returns a stream
                        measures = p.getElementsByClass('Measure')
                        if measures:
                            last_measure_num = max(last_measure_num, measures[-1].number)
                    except:
                        pass
            
            end_bass = self._get_measure_bass(score, last_measure_num) if last_measure_num > 0 else None

            if start_bass is None or end_bass is None:
                first_with_notes, last_with_notes = self._first_last_measure_with_notes(score)
                if start_bass is None and first_with_notes is not None:
                    start_bass = self._get_measure_bass(score, first_with_notes)
                if end_bass is None and last_with_notes is not None:
                    end_bass = self._get_measure_bass(score, last_with_notes)

            if start_bass is not None and end_bass is not None and start_bass == end_bass:
                heuristic_tonic = start_bass
                
                # If base_key is None or its tonic disagrees with our strong heuristic
                if base_key is None or base_key.tonic.pitchClass != heuristic_tonic:
                    # Use the heuristic tonic.
                    # Preserve mode from base_key if available, else default to 'major'
                    mode = (
                        base_key.mode
                        if base_key and base_key.mode in ["major", "minor"]
                        else "major"
                    )
                    
                    # Create new Key object
                    new_key = m21key.Key(tonic=m21pitch.Pitch(heuristic_tonic), mode=mode)
                    bass_override_key = new_key

        except Exception as e:
            # If heuristics fail, ignore and return m21_key
            pass

        final_key = bass_override_key or base_key
        if debug:
            m21_key_str = KeyStringNormalizer.normalize(m21_key)
            hist_key_str = KeyStringNormalizer.normalize(hist_key)
            base_key_str = KeyStringNormalizer.normalize(base_key)
            final_key_str = KeyStringNormalizer.normalize(final_key)
            m21_score_str = f"{m21_score:.3f}" if m21_score is not None else "NA"
            path_str = midi_path or "<unknown>"
            print(f"[KeyDetector] midi={path_str}")
            print(
                f"[KeyDetector] m21_key={m21_key_str} hist_key={hist_key_str} "
                f"hist_score={hist_score:.3f} m21_score={m21_score_str} "
                f"override_margin={self.hist_override_margin:.3f} "
                f"low_boost={self.low_pitch_boost:.2f} low_thresh={self.low_pitch_threshold}"
            )
            if raw_hist is not None:
                print(f"[KeyDetector] hist_top={self._top_pcs(hist)} raw_top={self._top_pcs(raw_hist)}")
            print(
                f"[KeyDetector] pre_bass={base_key_str} "
                f"bass_start={self._pc_name(start_bass)} "
                f"bass_end={self._pc_name(end_bass)} "
                f"final_key={final_key_str}"
            )

        return final_key

    def _get_measure_bass(self, score: stream.Score, measure_num: int) -> Optional[int]:
        """
        Returns the pitch class (0-11) of the lowest note in the specified measure number.
        """
        try:
            # score.measures returns a segment containing parts -> measures
            # We need to flatten to find all notes in that measure timeframe across all parts.
            measure_seg = score.measures(measure_num, measure_num)
            
            lowest = 128
            found = False
            
            for n in measure_seg.flatten().notes:
                if n.isNote:
                    if n.pitch.midi < lowest:
                        lowest = n.pitch.midi
                        found = True
                elif n.isChord:
                    for p in n.pitches:
                        if p.midi < lowest:
                            lowest = p.midi
                            found = True
                            
            return lowest % 12 if found else None
        except:
            return None

    def _detect_section_key(self, score: stream.Score, start_bar: int, end_bar: int) -> Optional[m21key.Key]:
        """
        section key를 안정적으로 잡기 위해 window 다수결(선택적)을 사용.
        안정성이 낮으면 None을 반환하여 상위에서 KEEP 처리하도록 유도.
        """
        # music21의 measures(a, b)는 보통 a~b inclusive처럼 동작하는 경우가 있어
        # 여기서는 보수적으로 window split 후 analyze 결과를 다수결로 본다.
        total_bars = max(0, end_bar - start_bar)
        if total_bars <= 0:
            return None

        windows = min(self.stability_windows, total_bars)
        if windows <= 1:
            return self._analyze_measures(score, start_bar, end_bar)

        size = total_bars // windows
        if size <= 0:
            return self._analyze_measures(score, start_bar, end_bar)

        votes: Dict[str, int] = {}
        for i in range(windows):
            ws = start_bar + i * size
            we = start_bar + (i + 1) * size if i < windows - 1 else end_bar
            k = self._analyze_measures(score, ws, we)
            ks = KeyStringNormalizer.normalize(k)
            if ks == "UNKNOWN":
                continue
            votes[ks] = votes.get(ks, 0) + 1

        if not votes:
            return None

        # 다수결이 뚜렷할 때만 채택 (동률이면 None)
        best = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        if len(best) >= 2 and best[0][1] == best[1][1]:
            return None

        # best[0] 문자열을 다시 Key로 되돌릴 필요는 없으므로,
        # 상위에서 문자열 비교를 하게 만들 수도 있지만, 여기서는 Key 객체 반환 형태 유지가 목적.
        # 간단히 다시 analyze를 전체 구간에서 수행해 반환.
        return self._analyze_measures(score, start_bar, end_bar)

    def _analyze_measures(self, score: stream.Score, start_bar: int, end_bar: int) -> Optional[m21key.Key]:
        # bar(0-based) → measure(1-based)로 변환 가정
        m1 = start_bar + 1
        m2 = end_bar  # end_bar exclusive → measure 상한은 구현체마다 다르므로 보수적으로 둠
        if m2 < m1:
            return None
        try:
            seg = score.measures(m1, m2)
            k = seg.analyze("key")
            if isinstance(k, m21key.Key):
                return k
            hist_key, _ = self._detect_key_by_histogram(seg)
            return hist_key
        except Exception:
            return None

    def _detect_key_by_histogram(self, score: stream.Score) -> Tuple[Optional[m21key.Key], float]:
        hist = self._pc_histogram(
            score,
            low_pitch_threshold=self.low_pitch_threshold,
            low_pitch_boost=self.low_pitch_boost,
        )
        if sum(hist) <= 0:
            return None, 0.0
        tonic_pc, mode, score_val = self._best_key_from_histogram(hist)
        if tonic_pc is None:
            return None, 0.0
        return m21key.Key(tonic=m21pitch.Pitch(tonic_pc), mode=mode), score_val

    def _pc_histogram(
        self,
        score: stream.Score,
        low_pitch_threshold: Optional[int] = None,
        low_pitch_boost: float = 0.0,
    ) -> List[float]:
        hist = [0.0] * 12
        for n in score.recurse().notes:
            try:
                if n.isNote:
                    if getattr(n, "isUnpitched", False):
                        continue
                    pc = n.pitch.pitchClass
                    weight = self._note_weight(n)
                    weight *= self._low_pitch_multiplier(
                        n.pitch.midi,
                        low_pitch_threshold,
                        low_pitch_boost,
                    )
                    hist[pc] += weight
                elif n.isChord:
                    weight = self._note_weight(n)
                    pitches = n.pitches
                    if not pitches:
                        continue
                    per_pitch = weight / len(pitches)
                    for p in pitches:
                        boost = self._low_pitch_multiplier(
                            p.midi,
                            low_pitch_threshold,
                            low_pitch_boost,
                        )
                        hist[p.pitchClass] += per_pitch * boost
            except Exception:
                continue
        return hist

    def _pc_histogram_pretty_midi(self, midi: pretty_midi.PrettyMIDI) -> List[float]:
        hist = [0.0] * 12
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                if note.end <= note.start:
                    continue
                dur = max(0.0, note.end - note.start)
                weight = max(0.0, dur) * max(0.5, float(note.velocity) / 64.0)
                weight *= self._low_pitch_multiplier(
                    note.pitch,
                    self.low_pitch_threshold,
                    self.low_pitch_boost,
                )
                hist[note.pitch % 12] += weight
        return hist

    def _best_key_by_diatonic_ratio(self, hist: List[float]) -> Tuple[Optional[int], str, float]:
        if sum(hist) <= 0:
            return None, "major", 0.0
        best_tonic = None
        best_mode = "major"
        best_ratio = -1.0
        best_profile = -1.0
        eps = 1e-6
        for tonic in range(12):
            for mode in ("major", "minor"):
                ratio = self._diatonic_ratio(hist, tonic, mode)
                if ratio > best_ratio + eps:
                    best_ratio = ratio
                    best_tonic = tonic
                    best_mode = mode
                    best_profile = self._key_profile_score(hist, tonic, mode)
                elif abs(ratio - best_ratio) <= eps:
                    profile = self._key_profile_score(hist, tonic, mode)
                    if profile > best_profile:
                        best_ratio = ratio
                        best_tonic = tonic
                        best_mode = mode
                        best_profile = profile
        return best_tonic, best_mode, best_ratio

    def _diatonic_ratio(self, hist: List[float], tonic_pc: int, mode: str) -> float:
        total = sum(hist)
        if total <= 0:
            return 0.0
        if mode == "minor":
            degrees = [0, 2, 3, 5, 7, 8, 10]
        else:
            degrees = [0, 2, 4, 5, 7, 9, 11]
        diatonic = sum(hist[(tonic_pc + d) % 12] for d in degrees)
        return diatonic / total

    def _note_weight(self, n) -> float:
        try:
            dur = float(n.duration.quarterLength) if n.duration else 1.0
        except Exception:
            dur = 1.0
        dur = max(0.0, dur)
        vel = None
        try:
            if n.volume is not None:
                vel = n.volume.velocity
        except Exception:
            vel = None
        if vel is None:
            return dur if dur > 0 else 1.0
        return max(dur, 0.0) * max(0.5, float(vel) / 64.0)

    def _best_key_from_histogram(self, hist: List[float]) -> Tuple[Optional[int], str, float]:
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        hist_norm = self._normalize_vec(hist)
        if not hist_norm:
            return None, "major", 0.0

        best_tonic = None
        best_mode = "major"
        best_score = -1.0
        for tonic in range(12):
            maj_score = self._key_profile_score(
                hist_norm,
                tonic,
                "major",
                major_profile,
                minor_profile,
            )
            if maj_score > best_score:
                best_score = maj_score
                best_tonic = tonic
                best_mode = "major"
            min_score = self._key_profile_score(
                hist_norm,
                tonic,
                "minor",
                major_profile,
                minor_profile,
            )
            if min_score > best_score:
                best_score = min_score
                best_tonic = tonic
                best_mode = "minor"

        return best_tonic, best_mode, best_score

    def _key_profile_score(
        self,
        hist_norm: List[float],
        tonic_pc: int,
        mode: str,
        major_profile: Optional[List[float]] = None,
        minor_profile: Optional[List[float]] = None,
    ) -> float:
        if major_profile is None:
            major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        if minor_profile is None:
            minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        hist = self._normalize_vec(hist_norm)
        if not hist:
            return 0.0
        profile = major_profile if mode == "major" else minor_profile
        rotated = self._rotate_profile(profile, tonic_pc)
        prof_norm = self._normalize_vec(rotated)
        if not prof_norm:
            return 0.0
        return sum(h * p for h, p in zip(hist, prof_norm))

    def _rotate_profile(self, profile: List[float], steps: int) -> List[float]:
        return [profile[(i - steps) % 12] for i in range(12)]

    def _normalize_vec(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return []
        return [v / norm for v in vec]

    def _pc_name(self, pc: Optional[int]) -> str:
        if pc is None:
            return "NA"
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return names[pc % 12]

    def _low_pitch_multiplier(
        self,
        pitch: int,
        threshold: Optional[int],
        boost: float,
    ) -> float:
        if threshold is None or boost <= 0:
            return 1.0
        return 1.0 + boost if pitch <= threshold else 1.0

    def _top_pcs(self, hist: List[float], top_n: int = 5) -> List[str]:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        ranked = sorted(range(12), key=lambda i: hist[i], reverse=True)
        return [f"{names[i]}:{hist[i]:.3f}" for i in ranked[:top_n]]

    def _first_last_measure_with_notes(self, score: stream.Score) -> Tuple[Optional[int], Optional[int]]:
        first = None
        last = None
        try:
            for p in score.parts:
                for m in p.getElementsByClass("Measure"):
                    if not m.notes:
                        continue
                    num = m.number
                    if first is None or num < first:
                        first = num
                    if last is None or num > last:
                        last = num
        except Exception:
            return None, None
        return first, last
