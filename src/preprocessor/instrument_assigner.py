class InstrumentRoleAssigner:
    def assign(self, midi) -> dict:
        """
        Rule-based 최소 구현
        """
        roles = {
            "MELODY": "PIANO",
            "HARMONY": "STRINGS",
            "BASS": "ELECTRIC_BASS",
            "DRUMS": "STANDARD_DRUMS",
        }
        return roles
