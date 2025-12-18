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
        dyn_low: float = 50.0,
        dyn_high: float = 90.0,
        den_sparse: float = 2.0,
        den_dense: float = 6.0,
        mov_threshold: float = 3.0,
        fill_density_jump: float = 0.5,
    ):
        self.dyn_low = dyn_low
        self.dyn_high = dyn_high
        self.den_sparse = den_sparse
        self.den_dense = den_dense
        self.mov_threshold = mov_threshold
        self.fill_density_jump = fill_density_jump

    def extract(self, midi, section: Section, analysis: Dict) -> Dict[str, str]:
        bars = analysis.get("bars", [])
        if not bars:
            return self._default_tokens()

        sec_bars = bars[section.start_bar : section.end_bar]
        if not sec_bars:
            return self._default_tokens()

        dyn = self._dyn(sec_bars)
        den = self._den(sec_bars)
        mov = self._mov(sec_bars)
        fill = self._fill(sec_bars)
        energy = self._energy(sec_bars)

        tokens = {
            "DYN": dyn,
            "DEN": den,
            "MOV": mov,
            "FILL": fill,
        }
        if energy is not None:
            tokens["ENERGY"] = str(energy)
        return tokens

    def _dyn(self, bars: List[Dict]) -> str:
        velocities = [b.get("mean_velocity") for b in bars if b.get("mean_velocity") is not None]
        if not velocities:
            return "MID"
        mean_vel = float(np.mean(velocities))
        if mean_vel < self.dyn_low:
            return "LOW"
        if mean_vel > self.dyn_high:
            return "HIGH"
        return "MID"

    def _den(self, bars: List[Dict]) -> str:
        densities = [b.get("note_count", 0) for b in bars]
        if not densities:
            return "NORMAL"
        mean_den = float(np.mean(densities))
        if mean_den < self.den_sparse:
            return "SPARSE"
        if mean_den > self.den_dense:
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

    def _energy(self, bars: List[Dict]) -> Optional[int]:
        # Simple 1-5 scale from mean note density and velocity
        densities = [b.get("note_count", 0) for b in bars]
        velocities = [b.get("mean_velocity") for b in bars if b.get("mean_velocity") is not None]
        if not densities:
            return None
        mean_den = float(np.mean(densities))
        mean_vel = float(np.mean(velocities)) if velocities else 64.0
        # Normalize roughly: density 0-12 -> 1-5, velocity 30-110 -> 1-5
        den_score = np.clip((mean_den / 12.0) * 4 + 1, 1, 5)
        vel_score = np.clip(((mean_vel - 30) / 80.0) * 4 + 1, 1, 5)
        energy = int(round((den_score + vel_score) / 2))
        return max(1, min(5, energy))

    def _default_tokens(self) -> Dict[str, str]:
        return {
            "DYN": "MID",
            "DEN": "NORMAL",
            "MOV": "STATIC",
            "FILL": "NO",
        }
