class DatasetBuilder:
    def build(self, midi_path: str, output_path: str):
        midi = MidiLoader().load(midi_path)
        analysis = MidiAnalyzer().analyze(midi)

        sections = StructureExtractor().extract_sections(midi, analysis)
        instruments = InstrumentRoleAssigner().assign(midi)

        prog_extractor = ChordProgressionExtractor()
        ctrl_extractor = ControlTokenExtractor()

        conductor_sections = ConductorTokenGenerator().generate(
            analysis, sections, prog_extractor, ctrl_extractor, midi
        )

        if not Validator().validate(conductor_sections):
            return

        DatasetExporter().export(
            analysis,
            instruments,
            conductor_sections,
            output_path
        )
