from collections import Counter

import numpy as np
import pretty_midi


class MidiAnalyzer:
    def analyze(self, midi: pretty_midi.PrettyMIDI) -> dict:
        """
        순수 분석 단계
        """
        tempos, tempi = midi.get_tempo_changes()
        global_bpm = int(np.median(tempi)) if len(tempi) else 120

        time_sigs = midi.time_signature_changes
        if time_sigs:
            ts = Counter((t.numerator, t.denominator) for t in time_sigs)
            numerator, denominator = ts.most_common(1)[0][0]
        else:
            numerator, denominator = 4, 4

        return {
            "global_bpm": global_bpm,
            "time_sig": (numerator, denominator),
        }
