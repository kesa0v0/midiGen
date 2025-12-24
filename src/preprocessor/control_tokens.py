from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from models import Section


class ControlTokenExtractor:
    """
    Derive macro control tokens from bar-level stats.
    """

    def __init__(
        self,
        mov_threshold: float = 3.0,
        fill_density_jump: float = 0.5,
        leap_low: float = 2.0,
        leap_high: float = 5.0,
        spacing_wide: float = 12.0,
    ):
        self.mov_threshold = mov_threshold
        self.fill_density_jump = fill_density_jump
        self.leap_low = leap_low
        self.leap_high = leap_high
        self.spacing_wide = spacing_wide

    def extract(self, midi, section: Section, analysis: Dict) -> Dict[str, str]:
        bars = analysis.get("bars", [])
        if not bars:
            return self._default_tokens()

        sec_bars = bars[section.start_bar : section.end_bar]
        if not sec_bars:
            return self._default_tokens()

        # Compute Global Stats for Relative Evaluation
        global_vels = [b.get("mean_velocity", 0) for b in bars if b.get("mean_velocity") is not None]
        global_dens = [b.get("note_count", 0) for b in bars]
        
        stats = {
            "mean_vel": float(np.mean(global_vels)) if global_vels else 64.0,
            "std_vel": float(np.std(global_vels)) if global_vels else 10.0,
            "mean_den": float(np.mean(global_dens)) if global_dens else 4.0,
            "std_den": float(np.std(global_dens)) if global_dens else 2.0,
        }

        dyn = self._dyn(sec_bars, stats)
        den = self._den(sec_bars, stats)
        mov = self._mov(sec_bars)
        fill = self._fill(sec_bars)
        energy = self._energy(sec_bars, bars)
        feel = self._feel(midi, sec_bars)
        leap = self._leap(midi, sec_bars)
        spacing = self._spacing(midi, sec_bars)

        tokens = {
            "DYN": dyn,
            "DEN": den,
            "MOV": mov,
            "FILL": fill,
            "FEEL": feel,
            "LEAP": leap,
            "SPACING": spacing,
        }
        if energy is not None:
            tokens["ENERGY"] = str(energy)
        return tokens

    def _dyn(self, bars: List[Dict], stats: Dict) -> str:
        velocities = [b.get("mean_velocity") for b in bars if b.get("mean_velocity") is not None]
        if not velocities:
            return "MID"
        mean_vel = float(np.mean(velocities))
        
        # Relative Thresholds: +/- 0.43 std dev (approx top/bottom 33% if normal dist)
        # Using 0.5 as requested for approx 30/40/30 split
        threshold_low = stats["mean_vel"] - 0.5 * stats["std_vel"]
        threshold_high = stats["mean_vel"] + 0.5 * stats["std_vel"]

        if mean_vel < threshold_low:
            return "LOW"
        if mean_vel > threshold_high:
            return "HIGH"
        return "MID"

    def _den(self, bars: List[Dict], stats: Dict) -> str:
        densities = [b.get("note_count", 0) for b in bars]
        if not densities:
            return "NORMAL"
        mean_den = float(np.mean(densities))
        
        threshold_sparse = stats["mean_den"] - 0.5 * stats["std_den"]
        threshold_dense = stats["mean_den"] + 0.5 * stats["std_den"]

        if mean_den < threshold_sparse:
            return "SPARSE"
        if mean_den > threshold_dense:
            return "DENSE"
        return "NORMAL"

    def _mov(self, bars: List[Dict]) -> str:
        centroids = [b.get("pitch_centroid") for b in bars if b.get("pitch_centroid") is not None]
        if len(centroids) < 2:
            return "STATIC"
        trend = np.polyfit(range(len(centroids)), centroids, 1)[0]
        if trend > self.mov_threshold:
            return "ASC"
        if trend < -self.mov_threshold:
            return "DESC"
        return "STATIC"

    def _fill(self, bars: List[Dict]) -> str:
        if len(bars) < 2:
            return "NO"
        last = bars[-1].get("note_count", 0)
        prev = bars[-2].get("note_count", 0)
        if prev == 0:
            return "NO"
        jump = (last - prev) / max(1, prev)
        return "YES" if jump >= self.fill_density_jump else "NO"

    def _energy(self, sec_bars: List[Dict], all_bars: List[Dict]) -> Optional[int]:
        # Calculate raw energy score for the current section
        def get_bar_score(bar):
            den = bar.get("note_count", 0)
            vel = bar.get("mean_velocity") or 64.0
            # Simple weighted sum: Density is often 0-16, Velocity 0-127. 
            # Scale density up to match velocity's influence roughly
            return vel + (den * 5.0)

        # 1. Calculate section average score
        sec_scores = [get_bar_score(b) for b in sec_bars]
        if not sec_scores:
            return 3
        sec_avg = float(np.mean(sec_scores))

        # 2. Calculate global min/max of BAR averages (smoothed) or Section averages?
        # Ideally we compare this section against all other sections, but we don't have section boundaries here easily.
        # We can approximate by comparing against the distribution of ALL bars.
        all_scores = [get_bar_score(b) for b in all_bars]
        if not all_scores:
            return 3
        
        global_min = float(np.min(all_scores))
        global_max = float(np.max(all_scores))
        
        if global_max - global_min < 1.0:
            return 3 # Flat dynamic

        # 3. Min-Max Normalize to 1-5
        normalized = (sec_avg - global_min) / (global_max - global_min)
        # Map 0.0-1.0 to 1-5
        energy = int(round(normalized * 4)) + 1
        return max(1, min(5, energy))

    def _feel(self, midi, sec_bars: List[Dict]) -> str:
        """
        Estimate swing vs straight by measuring 8th-note swing ratio within beats.
        """
        ratios = []
        notes = [
            n
            for inst in midi.instruments
            if not inst.is_drum
            for n in inst.notes
        ]
        if not notes:
            return "STRAIGHT"

        for bar in sec_bars:
            bpm = bar.get("bpm", 120.0)
            num, den = bar.get("time_sig", (4, 4))
            beat_dur = (60.0 / max(bpm, 1e-6)) * (4.0 / den)
            for b in range(int(num)):
                beat_start = bar["start"] + b * beat_dur
                beat_end = beat_start + beat_dur
                onsets = sorted(n.start for n in notes if beat_start <= n.start < beat_end)
                if len(onsets) < 2:
                    continue
                first, second = onsets[0], onsets[1]
                if second <= beat_start or second >= beat_end:
                    continue
                dur1 = second - beat_start
                dur2 = beat_end - second
                if dur1 <= 0 or dur2 <= 0:
                    continue
                ratio = dur1 / dur2
                ratios.append(ratio)

        if not ratios:
            return "STRAIGHT"

        med = float(np.median(ratios))
        if 1.5 <= med <= 2.5:
            return "SWING"
        return "STRAIGHT"

    def _leap(self, midi, sec_bars: List[Dict]) -> str:
        """
        Section-level leapiness: average melodic interval size.
        """
        notes = [
            n
            for inst in midi.instruments
            if not inst.is_drum
            for n in inst.notes
        ]
        if len(notes) < 2:
            return "LOW"
        notes_sorted = sorted(notes, key=lambda n: n.start)
        intervals = []
        for i in range(1, len(notes_sorted)):
            intervals.append(abs(notes_sorted[i].pitch - notes_sorted[i - 1].pitch))
        if not intervals:
            return "LOW"
        mean_int = float(np.mean(intervals))
        if mean_int <= self.leap_low:
            return "LOW"
        if mean_int <= self.leap_high:
            return "MID"
        return "HIGH"

    def _spacing(self, midi, sec_bars: List[Dict]) -> str:
        """
        Harmonic spacing hint: average pitch span within bars (non-drum notes).
        """
        spans = []
        for bar in sec_bars:
            start, end = bar["start"], bar["end"]
            pitches = [
                n.pitch
                for inst in midi.instruments
                if not inst.is_drum
                for n in inst.notes
                if start <= n.start < end
            ]
            if len(pitches) >= 2:
                spans.append(max(pitches) - min(pitches))
        if not spans:
            return "NARROW"
        mean_span = float(np.mean(spans))
        return "WIDE" if mean_span >= self.spacing_wide else "NARROW"

    def _default_tokens(self) -> Dict[str, str]:
        return {
            "DYN": "MID",
            "DEN": "NORMAL",
            "MOV": "STATIC",
            "FILL": "NO",
            "FEEL": "STRAIGHT",
            "LEAP": "LOW",
            "SPACING": "NARROW",
        }
