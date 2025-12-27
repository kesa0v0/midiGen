from typing import Optional

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
    def build(
        self, 
        midi_path: str, 
        output_path: str,
        genre: str = "UNKNOWN",
        style: str = "UNKNOWN",
        artist: str = "UNKNOWN",
        inst_type: str = "UNKNOWN",
        debug_key: bool = False,
        abs_chords: bool = False,
        adaptive_chord_grid: Optional[bool] = None,
        chord_detect_grid_unit: Optional[str] = None,
        chord_grid_unit: Optional[str] = None,
        chord_detail_mode: Optional[str] = None,
    ):
        midi_data = MidiLoader().load(midi_path)
        if midi_data is None:
            return

        midi = midi_data.midi
        if not PreValidator().validate(midi):
            print("[DatasetBuilder] Pre-validation failed.")
            return

        analysis = MidiAnalyzer().analyze(midi, midi_path)
        analysis["midi_type"] = midi_data.midi_type
        analysis["ticks_per_beat"] = midi_data.ticks_per_beat
        analysis["channel_programs"] = midi_data.channel_programs
        analysis["source_path"] = midi_path
        analysis["absolute_chords"] = abs_chords
        if adaptive_chord_grid is not None:
            analysis["adaptive_chord_grid"] = adaptive_chord_grid
        if chord_detect_grid_unit:
            analysis["chord_detect_grid_unit"] = chord_detect_grid_unit
        if chord_grid_unit:
            analysis["chord_grid_unit"] = chord_grid_unit
        if chord_detail_mode:
            analysis["chord_detail_mode"] = chord_detail_mode

        sections = StructureExtractor().extract_sections(midi, analysis)
        key_result = KeyDetector().detect(
            midi_path,
            [(sec.instance_id, sec.start_bar, sec.end_bar) for sec in sections],
            debug=debug_key,
            midi=midi,
        )
        analysis["global_key"] = key_result.global_key
        for sec in sections:
            key_val = key_result.section_keys.get(sec.instance_id)
            sec.local_key = None if key_val in (None, "KEEP") else key_val

        instruments, role_tracks = InstrumentRoleAssigner().assign_with_tracks(midi, inst_type=inst_type)
        analysis["role_tracks"] = role_tracks

        prog_extractor = ChordProgressionExtractor()
        ctrl_extractor = ControlTokenExtractor()

        conductor_bundle = ConductorTokenGenerator().generate(
            analysis, sections, prog_extractor, ctrl_extractor, midi
        )

        if not PostValidator().validate(conductor_bundle["global"], conductor_bundle["sections"]):
            print("[DatasetBuilder] Post-validation failed.")
            return

        DatasetExporter().export(
            conductor_bundle,
            instruments,
            output_path,
            midi_path,
            genre=genre,
            style=style,
            artist=artist,
            inst_type=inst_type
        )
