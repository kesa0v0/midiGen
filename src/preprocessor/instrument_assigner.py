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
