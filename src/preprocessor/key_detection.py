from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

from music21 import converter, stream, key as m21key


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
    def __init__(self, min_mod_bars: int = 4, stability_windows: int = 3):
        self.min_mod_bars = min_mod_bars
        self.stability_windows = stability_windows

    def detect(self, midi_path: str, sections: List[Tuple[str, int, int]]) -> KeyResult:
        """
        sections: list of (section_id, start_bar_inclusive, end_bar_exclusive)
        bar index는 0-based, end_bar는 exclusive로 가정.
        """
        score = converter.parse(midi_path)

        global_key = self._detect_global_key(score)
        global_key_str = KeyStringNormalizer.normalize(global_key)

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

    def _detect_global_key(self, score: stream.Score) -> Optional[m21key.Key]:
        # 1) 전체에서 시도
        try:
            k = score.analyze("key")
            if isinstance(k, m21key.Key):
                return k
        except Exception:
            pass

        # 2) 초반 일부(첫 16마디 근사)로 fallback
        try:
            # music21 measure는 보통 1-based. 여기서는 1~16 measure 사용 시도
            seg = score.measures(1, 16)
            k = seg.analyze("key")
            if isinstance(k, m21key.Key):
                return k
        except Exception:
            pass

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
            return k if isinstance(k, m21key.Key) else None
        except Exception:
            return None
