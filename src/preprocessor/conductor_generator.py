from models import ConductorSection, TimeSignature


class ConductorTokenGenerator:
    def generate(
        self,
        analysis,
        sections,
        prog_extractor,
        ctrl_extractor,
        midi
    ):
        conductor_sections = []

        for sec in sections:
            prog = prog_extractor.extract(midi, sec, analysis)
            ctrl = ctrl_extractor.extract(midi, sec, analysis)

            conductor_sections.append(
                ConductorSection(
                    name=sec.id,
                    bars=sec.end_bar - sec.start_bar,
                    bpm=None,
                    time_sig=TimeSignature(*analysis["time_sig"]),
                    key=None,
                    prog_grid=prog,
                    control_tokens=ctrl,
                )
            )

        return conductor_sections
