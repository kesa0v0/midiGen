from chord_progression import ChordProgressionExtractor
from conductor_generator import ConductorTokenGenerator
from control_tokens import ControlTokenExtractor
from exporter import DatasetExporter
from instrument_assigner import InstrumentRoleAssigner
from midi_analyzer import MidiAnalyzer
from midi_loader import MidiLoader
from structure_extractor import StructureExtractor
from validator import Validator


class DatasetBuilder:
    def build(self, midi_path: str, output_path: str):
        midi_data = MidiLoader().load(midi_path)
        if midi_data is None:
            return

        midi = midi_data.midi
        analysis = MidiAnalyzer().analyze(midi)
        analysis["midi_type"] = midi_data.midi_type
        analysis["ticks_per_beat"] = midi_data.ticks_per_beat
        analysis["channel_programs"] = midi_data.channel_programs

        sections = StructureExtractor().extract_sections(midi, analysis)
        instruments = InstrumentRoleAssigner().assign(midi)

        prog_extractor = ChordProgressionExtractor()
        ctrl_extractor = ControlTokenExtractor()

        conductor_bundle = ConductorTokenGenerator().generate(
            analysis, sections, prog_extractor, ctrl_extractor, midi
        )

        if not Validator().validate(conductor_bundle["sections"]):
            return

        DatasetExporter().export(
            conductor_bundle,
            instruments,
            output_path
        )
