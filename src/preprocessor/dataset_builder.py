from chord_progression import ChordProgressionExtractor
from conductor_generator import ConductorTokenGenerator
from control_tokens import ControlTokenExtractor
from exporter import DatasetExporter
from instrument_assigner import InstrumentRoleAssigner
from key_detection import KeyDetector
from midi_analyzer import MidiAnalyzer
from midi_loader import MidiLoader
from structure_extractor import StructureExtractor
from validator import PostValidator, PreValidator


class DatasetBuilder:
    def build(self, midi_path: str, output_path: str):
        midi_data = MidiLoader().load(midi_path)
        if midi_data is None:
            return

        midi = midi_data.midi
        if not PreValidator().validate(midi):
            return

        analysis = MidiAnalyzer().analyze(midi, midi_path)
        analysis["midi_type"] = midi_data.midi_type
        analysis["ticks_per_beat"] = midi_data.ticks_per_beat
        analysis["channel_programs"] = midi_data.channel_programs
        analysis["source_path"] = midi_path

        sections = StructureExtractor().extract_sections(midi, analysis)
        key_result = KeyDetector().detect(
            midi_path,
            [(sec.id, sec.start_bar, sec.end_bar) for sec in sections],
        )
        analysis["global_key"] = key_result.global_key
        for sec in sections:
            key_val = key_result.section_keys.get(sec.id)
            sec.local_key = None if key_val in (None, "KEEP") else key_val

        instruments = InstrumentRoleAssigner().assign(midi)

        prog_extractor = ChordProgressionExtractor()
        ctrl_extractor = ControlTokenExtractor()

        conductor_bundle = ConductorTokenGenerator().generate(
            analysis, sections, prog_extractor, ctrl_extractor, midi
        )

        if not PostValidator().validate(conductor_bundle["global"], conductor_bundle["sections"]):
            return

        DatasetExporter().export(
            conductor_bundle,
            instruments,
            output_path,
            midi_path,
        )
