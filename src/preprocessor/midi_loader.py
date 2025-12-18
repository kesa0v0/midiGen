from dataclasses import dataclass
from typing import Dict, Optional

import mido
import pretty_midi


@dataclass
class MidiData:
    midi: pretty_midi.PrettyMIDI
    midi_type: int
    ticks_per_beat: int
    channel_programs: Dict[int, int]


class MidiLoader:
    def load(self, path: str) -> Optional[MidiData]:
        """
        Load a MIDI file while preserving metadata.
        - catch file loading failures
        - capture type 0/1/2
        - capture ticks_per_beat (PPQ)
        - capture channel/program mapping
        """
        try:
            midi_file = mido.MidiFile(path)
        except Exception as exc:
            print(f"[MidiLoader] Failed to read MIDI file: {exc}")
            return None

        try:
            pm = pretty_midi.PrettyMIDI(midi_file=midi_file)
        except Exception as exc:
            print(f"[MidiLoader] Failed to parse MIDI with pretty_midi: {exc}")
            return None

        channel_programs: Dict[int, int] = {ch: 0 for ch in range(16)}
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == "program_change":
                    channel_programs[msg.channel] = msg.program

        return MidiData(
            midi=pm,
            midi_type=midi_file.type,
            ticks_per_beat=midi_file.ticks_per_beat,
            channel_programs=channel_programs,
        )
