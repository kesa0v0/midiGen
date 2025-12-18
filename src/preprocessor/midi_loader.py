import pretty_midi


class MidiLoader:
    def load(self, path: str) -> pretty_midi.PrettyMIDI:
        """
        MIDI 파일 로드.
        여기서는 절대 해석하지 않는다.
        """
        return pretty_midi.PrettyMIDI(path)
