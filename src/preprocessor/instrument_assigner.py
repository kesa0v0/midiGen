from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pretty_midi


@dataclass
class TrackStats:
    idx: int
    instrument: pretty_midi.Instrument
    note_count: int
    mean_pitch: Optional[float]
    min_pitch: Optional[int]
    max_pitch: Optional[int]
    pitch_range: Optional[int]
    mean_velocity: Optional[float]
    polyphony: float
    name_hint: str


class InstrumentRoleAssigner:
    """
    Heuristic (non-ML) role assignment with predictable fallbacks.
    """

    def assign(self, midi: pretty_midi.PrettyMIDI) -> Dict[str, str]:
        tracks = self._collect_tracks(midi)
        if not tracks:
            return {
                "MELODY": "UNKNOWN",
                "HARMONY": "UNKNOWN",
                "BASS": "UNKNOWN",
                "DRUMS": "UNKNOWN",
            }

        # Dynamic Strategy Selection
        non_drum_tracks = [t for t in tracks if not t.instrument.is_drum]
        has_drums = any(t.instrument.is_drum for t in tracks)

        # 1. Solo Instrument
        if len(non_drum_tracks) == 1:
            # If there are drums in a solo context (e.g. Piano + Drums), it's a duo, arguably "Band" logic is better?
            # User said: "Piano solo: No Drums, Bass tracks".
            # So if no drums and 1 track -> Solo.
            if not has_drums:
                return {"INSTRUMENT": self._instrument_name(non_drum_tracks[0])}
        
        # 2. Orchestra (Large Ensemble without Drum Kit)
        # Heuristic: No standard drum kit, high track count, or mostly orchestral instruments.
        # Simple threshold: No drums and >= 6 tracks.
        if not has_drums and len(non_drum_tracks) >= 6:
             return self._assign_orchestral(tracks)

        # 3. Pop/Band (Default)
        return self._assign_band(tracks)

    def _assign_band(self, tracks: List[TrackStats]) -> Dict[str, str]:
        drums = self._detect_drums(tracks)
        melody = self._pick_melody(tracks, drums)
        bass = self._pick_bass(tracks, drums)
        harmony = self._pick_harmony(tracks, drums, melody, bass)

        return {
            "MELODY": melody if melody else "UNKNOWN",
            "HARMONY": harmony if harmony else "UNKNOWN",
            "BASS": bass if bass else "UNKNOWN",
            "DRUMS": drums if drums else "UNKNOWN",
        }

    def _assign_orchestral(self, tracks: List[TrackStats]) -> Dict[str, str]:
        # Map GM programs to Families
        # Strings: 40-47 (Strings), 48-55 (Ensemble), 110 (Fiddle) -> 40-55, 110
        # Brass: 56-63
        # Woodwinds: 64-79 (Reed, Pipe)
        # Percussion: 8-15 (Chromatic Perc), 112-119 (Tinkle, Agogo...), 120-127 (Synth FX? No), Standard Drums
        # Keys/Others: Everything else
        
        families = {
            "STRINGS": [],
            "BRASS": [],
            "WOODWINDS": [],
            "PERCUSSION": []
        }

        for tr in tracks:
            prog = tr.instrument.program
            name = self._instrument_name(tr)
            
            if tr.instrument.is_drum:
                families["PERCUSSION"].append(name)
                continue
            
            # GM Mapping
            if 40 <= prog <= 55 or prog == 110:
                families["STRINGS"].append(name)
            elif 56 <= prog <= 63:
                families["BRASS"].append(name)
            elif 64 <= prog <= 79:
                families["WOODWINDS"].append(name)
            elif 8 <= prog <= 15 or 112 <= prog <= 119:
                families["PERCUSSION"].append(name)
            else:
                # Fallback for others (Piano, Guitar, Synth) -> Assign to most likely role or ignore?
                # For now, let's map Piano/Harp to Strings (common in orch) or Percussion?
                # Actually, Piano (0-7) is often used as Percussion or Strings equivalent in function.
                # Let's add a "KEYBOARD" or just map to Strings for Melody?
                # User request specifically asked for STRINGS, BRASS, WOODWINDS, PERCUSSION.
                # Let's map "Others" to the family they likely support or just pick the dominant one.
                # For simplicity, we can ignore or map to closest. 
                # Piano (0) -> Strings (often plays with strings).
                if 0 <= prog <= 7:
                    families["STRINGS"].append(name) # Piano treated as Strings/Keys layer
                elif 24 <= prog <= 39: # Guitar/Bass
                    families["STRINGS"].append(name) # Plucked strings
                else:
                    families["WOODWINDS"].append(name) # Synth/Ethnic -> Woodwinds buffer

        # Select representative for each family (e.g. most notes or highest polyphony)
        result = {}
        for fam, candidates in families.items():
            if not candidates:
                result[fam] = "NONE"
            else:
                # Simple heuristic: Just pick the first unique one, or join them?
                # "STRINGS=VIOLIN" is better than "STRINGS=VIOLIN,CELLO" for token consistency?
                # Let's pick the most frequent one?
                # Since we stored names, we lost the stats. 
                # Ideally we should select based on stats.
                # But for now, returning the most common name in that family from the track list is OK.
                # Let's just return the primary one (first found or most frequent).
                # To be deterministic, sort and pick first?
                # Or count frequency.
                from collections import Counter
                most_common = Counter(candidates).most_common(1)[0][0]
                result[fam] = most_common
        
        return result

    # ---- Track stats ----
    def _collect_tracks(self, midi: pretty_midi.PrettyMIDI) -> List[TrackStats]:
        tracks: List[TrackStats] = []
        for idx, inst in enumerate(midi.instruments):
            if not inst.notes and not inst.is_drum:
                continue
            note_count = len(inst.notes)
            if note_count:
                pitches = np.array([n.pitch for n in inst.notes])
                velocities = np.array([n.velocity for n in inst.notes])
                min_pitch = int(pitches.min())
                max_pitch = int(pitches.max())
                mean_pitch = float(pitches.mean())
                pitch_range = int(max_pitch - min_pitch)
                mean_velocity = float(velocities.mean())
                polyphony = self._estimate_polyphony(inst.notes)
            else:
                min_pitch = max_pitch = pitch_range = None
                mean_pitch = mean_velocity = None
                polyphony = 0.0

            tracks.append(
                TrackStats(
                    idx=idx,
                    instrument=inst,
                    note_count=note_count,
                    mean_pitch=mean_pitch,
                    min_pitch=min_pitch,
                    max_pitch=max_pitch,
                    pitch_range=pitch_range,
                    mean_velocity=mean_velocity,
                    polyphony=polyphony,
                    name_hint=inst.name or "",
                )
            )
        return tracks

    def _estimate_polyphony(self, notes: List[pretty_midi.Note]) -> float:
        events = []
        for n in notes:
            events.append((n.start, 1))
            events.append((n.end, -1))
        if not events:
            return 0.0
        events.sort(key=lambda x: (x[0], -x[1]))
        active = 0
        total = 0.0
        last_t = events[0][0]
        for t, delta in events:
            duration = t - last_t
            if duration > 0:
                total += active * duration
            active += delta
            last_t = t
        total_time = events[-1][0] - events[0][0]
        if total_time <= 0:
            return float(active)
        return total / total_time

    # ---- Role pickers ----
    def _detect_drums(self, tracks: List[TrackStats]) -> Optional[str]:
        for tr in tracks:
            if tr.instrument.is_drum:
                return "STANDARD_DRUMS"
            if "DRUM" in tr.name_hint.upper() or "PERC" in tr.name_hint.upper():
                return "STANDARD_DRUMS"
        return None

    def _pick_melody(self, tracks: List[TrackStats], drums: Optional[str]) -> Optional[str]:
        candidates = []
        for tr in tracks:
            if tr.instrument.is_drum:
                continue
            if tr.note_count == 0:
                continue
            monophony_bias = max(0.0, 1.5 - tr.polyphony)
            score = (tr.mean_pitch or 0) + 5 * monophony_bias + 0.01 * tr.note_count
            name_bonus = self._name_bonus(tr.name_hint, ["LEAD", "MELODY", "VOCAL", "VOICE"])
            score += name_bonus
            candidates.append((score, tr))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return self._instrument_name(candidates[0][1])

    def _pick_bass(self, tracks: List[TrackStats], drums: Optional[str]) -> Optional[str]:
        candidates = []
        for tr in tracks:
            if tr.instrument.is_drum or tr.note_count == 0:
                continue
            score = 0.0
            if tr.mean_pitch is not None:
                score -= tr.mean_pitch
            if tr.max_pitch is not None:
                score -= 0.5 * tr.max_pitch
            score += self._name_bonus(tr.name_hint, ["BASS"])
            candidates.append((score, tr))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return self._instrument_name(candidates[0][1])

    def _pick_harmony(
        self,
        tracks: List[TrackStats],
        drums: Optional[str],
        melody: Optional[str],
        bass: Optional[str],
    ) -> Optional[str]:
        used_names = {melody, bass, drums}
        candidates = []
        for tr in tracks:
            inst_name = self._instrument_name(tr)
            if inst_name in used_names:
                continue
            if tr.instrument.is_drum or tr.note_count == 0:
                continue
            poly_score = min(tr.polyphony, 4.0)
            range_score = (tr.pitch_range or 0) * 0.1
            score = poly_score + range_score + 0.005 * tr.note_count
            score += self._name_bonus(tr.name_hint, ["PAD", "STR", "CHORD", "COMP"])
            candidates.append((score, tr))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return self._instrument_name(candidates[0][1])

    # ---- Helpers ----
    def _name_bonus(self, name: str, keywords: List[str]) -> float:
        upper = name.upper()
        return 10.0 if any(kw in upper for kw in keywords) else 0.0

    def _instrument_name(self, stats: TrackStats) -> str:
        if stats.instrument.is_drum:
            return "STANDARD_DRUMS"
        program_name = pretty_midi.program_to_instrument_name(stats.instrument.program)
        return self._normalize_name(stats.name_hint) or self._normalize_name(program_name) or "UNKNOWN"

    def _normalize_name(self, name: str) -> str:
        if not name:
            return ""
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.upper()).strip("_")
        return cleaned or ""
