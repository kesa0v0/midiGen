class ControlTokenExtractor:
    def extract(self, midi, section: Section) -> dict:
        """
        통계 기반 매크로 컨트롤
        """
        return {
            "DYN": "MID",
            "DEN": "NORMAL",
            "MOV": "STATIC",
            "FILL": "NO",
        }
