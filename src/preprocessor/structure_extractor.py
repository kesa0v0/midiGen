from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from models import Section, TimeSignature


class StructureExtractor:
    """
    Heuristic section splitter (no ML).
    - Split on BPM / TIME_SIG change
    - Split on abrupt bar-stat changes
    - Split at detected repeating bar patterns
    - Enforce a minimum section length
    """

    def __init__(
        self,
        min_section_bars: int = 4,
        density_jump: float = 0.6,
        velocity_jump: float = 20.0,
        centroid_jump: float = 5.0,
    ):
        self.min_section_bars = min_section_bars
        self.density_jump = density_jump
        self.velocity_jump = velocity_jump
        self.centroid_jump = centroid_jump

    def extract_sections(self, midi, analysis) -> List[Section]:
        bars = analysis.get("bars", [])
        tempo_map = analysis.get("tempo_map", [])
        time_sig_map = analysis.get("time_sig_map", [])

        if not bars:
            return [
                Section(
                    id="A",
                    start_bar=0,
                    end_bar=0,
                )
            ]

        total_bars = len(bars)
        breakpoints = set([0, total_bars])

        # BPM change points -> bar boundary
        for ev in tempo_map[1:]:
            idx = self._bar_index_for_time(bars, ev["time"])
            if idx is not None:
                breakpoints.add(idx)

        # TIME_SIG change points -> bar boundary
        for ev in time_sig_map[1:]:
            idx = self._bar_index_for_time(bars, ev["time"])
            if idx is not None:
                breakpoints.add(idx)

        # Abrupt bar-stat change detection
        for i in range(1, total_bars):
            if self._is_stat_change(bars[i - 1], bars[i]):
                breakpoints.add(i)

        # Repeating pattern detection (look for immediate repetition of 8 or 4-bar blocks)
        fingerprints = [self._bar_fingerprint(b) for b in bars]
        for size in (8, 4):
            for i in range(size, total_bars - size + 1):
                if fingerprints[i - size : i] == fingerprints[i : i + size]:
                    breakpoints.add(i)

        # Enforce minimum section length by pruning short gaps
        sorted_pts = sorted(breakpoints)
        filtered = [sorted_pts[0]]
        for pt in sorted_pts[1:]:
            if pt - filtered[-1] < self.min_section_bars:
                continue
            filtered.append(pt)
        if filtered[-1] != total_bars:
            filtered.append(total_bars)
        # If trailing tail is too short, merge into previous
        if len(filtered) >= 2 and (filtered[-1] - filtered[-2]) < self.min_section_bars:
            filtered.pop(-2)

        spans = []
        for idx in range(len(filtered) - 1):
            spans.append((filtered[idx], filtered[idx + 1]))

        labels = self._label_sections(bars, spans)

        sections: List[Section] = []
        for idx, (start, end) in enumerate(spans):
            if start >= end:
                continue

            first_bar = bars[start]
            local_ts = self._consistent_time_sig(bars[start:end])
            local_bpm = self._consistent_bpm(bars[start:end])
            local_key = self._section_key_candidate(bars[start:end])

            sections.append(
                Section(
                    id=labels[idx],
                    start_bar=start,
                    end_bar=end,
                    local_bpm=local_bpm,
                    local_time_sig=local_ts,
                    local_key=local_key,
                )
            )

        return sections

    def _bar_index_for_time(self, bars: List[Dict], time: float) -> Optional[int]:
        for i, bar in enumerate(bars):
            if bar["start"] >= time:
                return i
        return None

    def _is_stat_change(self, prev: Dict, curr: Dict) -> bool:
        # note density change
        p_notes, c_notes = prev.get("note_count", 0), curr.get("note_count", 0)
        if p_notes > 0 and abs(c_notes - p_notes) / max(1, p_notes) >= self.density_jump:
            return True
        if p_notes == 0 and c_notes >= 4:
            return True

        # velocity jump
        pv, cv = prev.get("mean_velocity"), curr.get("mean_velocity")
        if pv is not None and cv is not None and abs(cv - pv) >= self.velocity_jump:
            return True

        # pitch centroid jump
        pc_prev, pc_curr = prev.get("pitch_centroid"), curr.get("pitch_centroid")
        if pc_prev is not None and pc_curr is not None and abs(pc_curr - pc_prev) >= self.centroid_jump:
            return True

        # key change
        k_prev, k_curr = prev.get("key_candidate"), curr.get("key_candidate")
        if k_prev not in (None, "UNKNOWN") and k_curr not in (None, "UNKNOWN") and k_prev != k_curr:
            return True

        return False

    def _bar_fingerprint(self, bar: Dict) -> Tuple[int, int, int, str]:
        # Coarse quantization to detect repeated patterns
        density_bin = min(9, bar.get("note_count", 0) // 4)
        vel = bar.get("mean_velocity")
        velocity_bin = 0 if vel is None else int(vel // 16)
        pc = bar.get("pitch_centroid")
        pitch_bin = 0 if pc is None else int(pc // 4)
        key = bar.get("key_candidate", "UNKNOWN")
        return (density_bin, velocity_bin, pitch_bin, key)

    def _section_key_candidate(self, bars: List[Dict]) -> Optional[str]:
        keys = [b.get("key_candidate") for b in bars if b.get("key_candidate") not in (None, "UNKNOWN")]
        if not keys:
            return None
        most = Counter(keys).most_common(1)[0][0]
        return most

    def _label_sections(self, bars: List[Dict], spans: List[tuple]) -> List[str]:
        """
        Heuristic labels: INTRO, VERSE, CHORUS, BRIDGE, OUTRO with numeric suffix if repeated.
        """
        if not spans:
            return []

        densities = []
        velocities = []
        for start, end in spans:
            seg = bars[start:end]
            dens = [b.get("note_count", 0) for b in seg]
            vels = [b.get("mean_velocity") for b in seg if b.get("mean_velocity") is not None]
            densities.append(float(sum(dens)) / max(1, len(dens)))
            velocities.append(float(sum(vels)) / max(1, len(vels)) if vels else 0.0)

        scores = [d + v * 0.1 for d, v in zip(densities, velocities)]
        max_idx = int(np.argmax(scores)) if scores else 0
        min_idx = int(np.argmin(scores)) if scores else 0

        labels = []
        counts = {"INTRO": 0, "VERSE": 0, "CHORUS": 0, "BRIDGE": 0, "OUTRO": 0}
        for i, (start, end) in enumerate(spans):
            bars_len = end - start
            if i == 0:
                label = "INTRO"
            elif i == len(spans) - 1:
                label = "OUTRO"
            elif i == max_idx:
                label = "CHORUS"
            elif i == min_idx and bars_len >= 4:
                label = "BRIDGE"
            else:
                label = "VERSE"
            counts[label] += 1
            suffix = f"_{counts[label]}" if counts[label] > 1 else ""
            labels.append(label + suffix)
        return labels

    def _consistent_time_sig(self, bars: List[Dict]) -> Optional[TimeSignature]:
        if not bars:
            return None
        ts_set = {(b.get("time_sig")[0], b.get("time_sig")[1]) for b in bars if b.get("time_sig")}
        if len(ts_set) == 1:
            num, den = next(iter(ts_set))
            return TimeSignature(num, den)
        return None

    def _consistent_bpm(self, bars: List[Dict]) -> Optional[int]:
        if not bars:
            return None
        bpms = {int(round(b.get("bpm"))) for b in bars if b.get("bpm") is not None}
        if len(bpms) == 1:
            return bpms.pop()
        return None

    def _time_sig_from_bar(self, bar: Dict) -> Optional[TimeSignature]:
        ts = bar.get("time_sig")
        if not ts:
            return None
        return TimeSignature(ts[0], ts[1])
