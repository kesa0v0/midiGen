from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi


@dataclass
class BarInfo:
    index: int
    start: float
    end: float
    bpm: float
    time_sig: Tuple[int, int]
    note_count: int
    mean_velocity: Optional[float]
    pitch_centroid: Optional[float]
    key_candidate: str


class MidiAnalyzer:
    def analyze(self, midi: pretty_midi.PrettyMIDI) -> dict:
        """
        Tempo/time-signature map + bar-wise statistics.
        """
        tempos, tempi = midi.get_tempo_changes()
        tempo_map = self._build_tempo_map(tempos, tempi)
        global_bpm = int(np.median(tempi)) if len(tempi) else 120

        time_sig_map = self._build_time_sig_map(midi)
        numerator, denominator = self._most_common_time_sig(time_sig_map)

        bars = self._compute_bars(midi, tempo_map, time_sig_map)
        bar_stats = self._compute_bar_stats(midi, bars)

        return {
            "global_bpm": global_bpm,
            "time_sig": (numerator, denominator),
            "tempo_map": tempo_map,
            "time_sig_map": time_sig_map,
            "bars": bar_stats,
        }

    def _build_tempo_map(self, times: np.ndarray, tempi: np.ndarray) -> List[Dict[str, float]]:
        if len(times) == 0:
            return [{"time": 0.0, "bpm": 120.0}]
        return [{"time": float(t), "bpm": float(b)} for t, b in zip(times, tempi)]

    def _build_time_sig_map(self, midi: pretty_midi.PrettyMIDI) -> List[Dict[str, float]]:
        changes = midi.time_signature_changes
        if not changes:
            return [{"time": 0.0, "numerator": 4, "denominator": 4}]
        return [
            {
                "time": float(ts.time),
                "numerator": int(ts.numerator),
                "denominator": int(ts.denominator),
            }
            for ts in changes
        ]

    def _most_common_time_sig(self, time_sig_map: List[Dict[str, float]]) -> Tuple[int, int]:
        ts = Counter((ts["numerator"], ts["denominator"]) for ts in time_sig_map)
        return ts.most_common(1)[0][0]

    def _compute_bars(
        self,
        midi: pretty_midi.PrettyMIDI,
        tempo_map: List[Dict[str, float]],
        time_sig_map: List[Dict[str, float]],
    ) -> List[BarInfo]:
        downbeats = list(midi.get_downbeats())
        if not downbeats or downbeats[0] > 0.0:
            downbeats = [0.0] + downbeats

        end_time = midi.get_end_time()
        if not downbeats or downbeats[-1] < end_time:
            downbeats.append(end_time)

        tempo_map = sorted(tempo_map, key=lambda x: x["time"])
        time_sig_map = sorted(time_sig_map, key=lambda x: x["time"])

        def latest_event(time: float, events: List[Dict[str, float]], key_fields: Tuple[str, ...]):
            last = events[0]
            for ev in events:
                if ev["time"] <= time:
                    last = ev
                else:
                    break
            return tuple(last[k] for k in key_fields)

        bars: List[BarInfo] = []
        for idx in range(len(downbeats) - 1):
            start = float(downbeats[idx])
            end = float(downbeats[idx + 1])
            bpm = latest_event(start, tempo_map, ("bpm",))[0]
            num, den = latest_event(start, time_sig_map, ("numerator", "denominator"))
            bars.append(
                BarInfo(
                    index=idx,
                    start=start,
                    end=end,
                    bpm=bpm,
                    time_sig=(num, den),
                    note_count=0,
                    mean_velocity=None,
                    pitch_centroid=None,
                    key_candidate="UNKNOWN",
                )
            )
        return bars

    def _compute_bar_stats(self, midi: pretty_midi.PrettyMIDI, bars: List[BarInfo]) -> List[Dict]:
        if not bars:
            return []

        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        bar_stats: List[Dict] = []
        for bar in bars:
            notes = self._notes_in_bar(midi, bar.start, bar.end)
            if notes:
                velocities = np.array([n.velocity for n in notes])
                pitches = np.array([n.pitch for n in notes])
                weights = velocities.astype(float)
                mean_velocity = float(np.mean(velocities))
                pitch_centroid = float(np.average(pitches, weights=weights)) if weights.sum() > 0 else float(np.mean(pitches))
                key_candidate = self._estimate_bar_key(notes, major_profile, minor_profile, bar.start, bar.end)
            else:
                mean_velocity = None
                pitch_centroid = None
                key_candidate = "UNKNOWN"

            bar_stats.append(
                {
                    "index": bar.index,
                    "start": bar.start,
                    "end": bar.end,
                    "bpm": bar.bpm,
                    "time_sig": bar.time_sig,
                    "note_count": len(notes),
                    "mean_velocity": mean_velocity,
                    "pitch_centroid": pitch_centroid,
                    "key_candidate": key_candidate,
                }
            )
        return bar_stats

    def _notes_in_bar(self, midi: pretty_midi.PrettyMIDI, start: float, end: float):
        collected = []
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                if start <= note.start < end:
                    collected.append(note)
        return collected

    def _estimate_bar_key(
        self,
        notes,
        major_profile: np.ndarray,
        minor_profile: np.ndarray,
        start: float,
        end: float,
    ) -> str:
        if not notes:
            return "UNKNOWN"

        pc_hist = np.zeros(12, dtype=float)
        for n in notes:
            overlap = max(0.0, min(n.end, end) - max(n.start, start))
            weight = overlap if overlap > 0 else 0.0
            weight *= n.velocity
            pc_hist[n.pitch % 12] += weight if weight > 0 else n.velocity

        if pc_hist.sum() == 0:
            return "UNKNOWN"

        def score(profile):
            best_corr = None
            best_pc = None
            for shift in range(12):
                rotated = np.roll(profile, shift)
                corr = np.corrcoef(pc_hist, rotated)[0, 1]
                if best_corr is None or corr > best_corr:
                    best_corr = corr
                    best_pc = shift
            return best_corr, best_pc

        major_corr, major_pc = score(major_profile)
        minor_corr, minor_pc = score(minor_profile)

        if major_corr is None or minor_corr is None:
            return "UNKNOWN"

        if major_corr >= minor_corr:
            tonic_pc = major_pc
            mode = "MAJOR"
        else:
            tonic_pc = minor_pc
            mode = "MINOR"

        pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        tonic = pitch_names[tonic_pc % 12]
        return f"{tonic}_{mode}"
