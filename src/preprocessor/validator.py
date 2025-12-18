class Validator:
    def validate(self, sections) -> bool:
        """
        실패 시 False → 데이터 폐기
        """
        for sec in sections:
            if sec.bars != len(sec.prog_grid):
                return False
        return True
