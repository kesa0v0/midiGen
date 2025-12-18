from typing import List
from models import Section


class StructureExtractor:
    def extract_sections(self, midi, analysis) -> List[Section]:
        """
        최소 구현:
        - BPM / TIME_SIG 변경이 없으면 전체를 하나의 SECTION
        - 이후 점진적으로 휴리스틱 추가 가능
        """
        total_bars = int(midi.get_end_time() // 2)  # rough
        return [
            Section(
                id="A",
                start_bar=0,
                end_bar=total_bars,
            )
        ]
