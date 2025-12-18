class ChordProgressionExtractor:
    def extract(self, midi, section: Section) -> list:
        """
        매우 보수적인 기본 구현
        - bar 단위로 코드 하나
        - 지금은 전부 I 로 채움 (placeholder)
        """
        bars = section.end_bar - section.start_bar
        prog = []
        for _ in range(bars):
            prog.append(["I", "-", "-", "-"])
        return prog
