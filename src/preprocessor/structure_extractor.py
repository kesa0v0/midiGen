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
        min_section_bars: int = 8,
        density_jump: float = 0.6,
        velocity_jump: float = 20.0,
        centroid_jump: float = 5.0,
        stat_persistence: int = 2,
    ):
        self.min_section_bars = min_section_bars
        self.density_jump = density_jump
        self.velocity_jump = velocity_jump
        self.centroid_jump = centroid_jump
        self.stat_persistence = max(1, stat_persistence)

    def extract_sections(self, midi, analysis) -> List[Section]:
        bars = analysis.get("bars", [])
        tempo_map = analysis.get("tempo_map", [])
        time_sig_map = analysis.get("time_sig_map", [])

        if not bars:
            return [
                Section(
                    id="SECTION_A",
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

        # Abrupt bar-stat change detection (softened: require consecutive signals)
        stat_streak = 0
        for i in range(1, total_bars):
            if self._is_stat_change(bars[i - 1], bars[i]):
                stat_streak += 1
                if stat_streak >= self.stat_persistence:
                    breakpoints.add(i)
            else:
                stat_streak = 0

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
        spans = self._merge_similar_spans(spans, bars)

        labeled_info = self._label_sections(bars, spans)

        sections: List[Section] = []
        for idx, (start, end) in enumerate(spans):
            if start >= end:
                continue
            
            label, role = labeled_info[idx]

            first_bar = bars[start]
            local_ts = self._consistent_time_sig(bars[start:end])
            local_bpm = self._consistent_bpm(bars[start:end])
            local_key = self._section_key_candidate(bars[start:end])

            sections.append(
                Section(
                    id=label,
                    start_bar=start,
                    end_bar=end,
                    local_bpm=local_bpm,
                    local_time_sig=local_ts,
                    local_key=local_key,
                    role=role
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

    def _label_sections(self, bars: List[Dict], spans: List[tuple]) -> List[Tuple[str, Optional[str]]]:
        """
        Abstract labeling: SECTION_A, SECTION_B, ...
        Groups similar sections together.
        Determines 'MAIN_THEME' based on energy and repetition.
        """
        if not spans:
            return []

        # 1. Compute stats for each span
        span_stats = []
        for start, end in spans:
            seg = bars[start:end]
            dens = [b.get("note_count", 0) for b in seg]
            vels = [b.get("mean_velocity") for b in seg if b.get("mean_velocity") is not None]
            
            avg_dens = float(sum(dens)) / max(1, len(dens))
            avg_vel = float(sum(vels)) / max(1, len(vels)) if vels else 0.0
            energy = avg_dens + avg_vel * 0.1
            
            span_stats.append({
                "avg_dens": avg_dens,
                "avg_vel": avg_vel,
                "energy": energy,
                "start": start,
                "end": end
            })

        # 2. Cluster sections (Types)
        # types: list of {"stats": stats, "label": "SECTION_X"}
        types = []
        labels = [] # Index in 'types' for each span

        for i, stats in enumerate(span_stats):
            match_idx = -1
            # Check similarity with existing types
            for t_idx, t_data in enumerate(types):
                ref = t_data["stats"]
                # Use strict thresholds similar to _similar_sections but perhaps slightly looser for global matching?
                # Using the same logic as _similar_sections for consistency
                dens_close = abs(stats["avg_dens"] - ref["avg_dens"]) <= 2.5 # slightly looser than 2.0
                vel_close = abs(stats["avg_vel"] - ref["avg_vel"]) <= 20.0   # slightly looser than 15.0
                
                # Note: Key/TimeSig are not checked here, purely based on energy/density for structural type
                if dens_close and vel_close:
                    match_idx = t_idx
                    break
            
            if match_idx != -1:
                labels.append(match_idx)
                types[match_idx]["count"] += 1
                # Update stats average? (Optional, skipping for stability)
            else:
                # Create new type
                new_label_char = chr(ord('A') + len(types))
                # Fallback if we run out of letters (unlikely for typical song)
                if len(types) >= 26:
                    new_label_char = f"Z{len(types)}"
                
                types.append({
                    "stats": stats,
                    "label": f"SECTION_{new_label_char}",
                    "count": 1,
                    "id": len(types)
                })
                labels.append(len(types) - 1)

        # 3. Identify MAIN_THEME
        # Criteria: Repeated (count > 1) AND Highest Energy among repeated sections
        max_energy = -1.0
        main_theme_idx = -1

        for i, t_data in enumerate(types):
            if t_data["count"] > 1:
                if t_data["stats"]["energy"] > max_energy:
                    max_energy = t_data["stats"]["energy"]
                    main_theme_idx = i
        
        # If no section repeats, maybe pick the highest energy one anyway? 
        # Or strictly follow "repeats" rule. Pop songs usually have repeating Chorus.
        # If nothing repeats, we might be in a linear progression. No Main Theme?
        # Let's enforce repetition for MAIN_THEME to be safe.

        # 4. Generate Output (Label, Role)
        result = []
        for idx in labels:
            label_str = types[idx]["label"]
            role = None
            if idx == main_theme_idx:
                role = "MAIN_THEME"
            result.append((label_str, role))

        return result


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

    def _merge_similar_spans(self, spans: List[tuple], bars: List[Dict]) -> List[tuple]:
        """
        Merge adjacent spans if their stats/time/key are similar to avoid over-splitting.
        """
        if not spans:
            return []

        def span_stats(start, end):
            slice_bars = bars[start:end]
            dens = np.mean([b.get("note_count", 0) for b in slice_bars]) if slice_bars else 0
            vel_vals = [b.get("mean_velocity") for b in slice_bars if b.get("mean_velocity") is not None]
            vel = np.mean(vel_vals) if vel_vals else 0
            key = self._section_key_candidate(slice_bars)
            ts = self._consistent_time_sig(slice_bars)
            bpm = self._consistent_bpm(slice_bars)
            return dens, vel, key, ts, bpm

        merged = [spans[0]]
        prev_stats = span_stats(*spans[0])

        for sp in spans[1:]:
            cur_stats = span_stats(*sp)
            if self._similar_sections(prev_stats, cur_stats):
                # merge into previous
                start = merged[-1][0]
                merged[-1] = (start, sp[1])
                prev_stats = span_stats(start, sp[1])
            else:
                merged.append(sp)
                prev_stats = cur_stats

        return merged

    def _similar_sections(self, s1, s2) -> bool:
        dens1, vel1, key1, ts1, bpm1 = s1
        dens2, vel2, key2, ts2, bpm2 = s2

        dens_close = abs(dens1 - dens2) <= 2.0
        vel_close = abs(vel1 - vel2) <= 15.0
        key_ok = (key1 is None or key1 == "UNKNOWN" or key2 is None or key2 == "UNKNOWN" or key1 == key2)
        ts_ok = (ts1 is None or ts2 is None or (ts1.numerator == ts2.numerator and ts1.denominator == ts2.denominator))
        bpm_ok = (bpm1 is None or bpm2 is None or abs(bpm1 - bpm2) <= 5)
        return dens_close and vel_close and key_ok and ts_ok and bpm_ok
