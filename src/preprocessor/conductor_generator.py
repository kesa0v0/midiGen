from models import ConductorSection, TimeSignature


class ConductorTokenGenerator:
    def __init__(self, default_grid_unit: str = "1/8"):
        self.default_grid_unit = default_grid_unit

    def generate(
        self,
        analysis,
        sections,
        prog_extractor,
        ctrl_extractor,
        midi,
    ):
        """
        Assemble conductor tokens with GLOBAL defaults and SECTION overrides.
        """
        global_bpm, global_ts = self._global_defaults(analysis)
        global_key = analysis.get("global_key", None)
        grid_unit = self._grid_unit(analysis, global_ts)

        conductor_sections = []

        for sec in sections:
            local_bpm = sec.local_bpm or global_bpm
            local_ts = sec.local_time_sig or TimeSignature(*global_ts)
            local_key = sec.local_key or global_key

            prog = prog_extractor.extract(midi, sec, analysis)
            ctrl = ctrl_extractor.extract(midi, sec, analysis)

            conductor_sections.append(
                ConductorSection(
                    name=sec.id,
                    bars=sec.end_bar - sec.start_bar,
                    bpm=local_bpm,
                    time_sig=local_ts,
                    key=local_key,
                    prog_grid=prog,
                    control_tokens=ctrl,
                )
            )

        form = self._build_form(sections)

        return {
            "global": {
                "bpm": global_bpm,
                "time_sig": global_ts,
                "key": global_key,
                "grid_unit": grid_unit,
            },
            "form": form,
            "sections": conductor_sections,
        }

    def _global_defaults(self, analysis):
        bpm = analysis.get("global_bpm", 120)
        ts = analysis.get("time_sig", (4, 4))
        return bpm, ts

    def _grid_unit(self, analysis, global_ts):
        # If analysis contains a suggested grid, use it; otherwise set from denom
        if "grid_unit" in analysis:
            return analysis["grid_unit"]
        _, den = global_ts
        if den >= 8:
            return "1/8"
        return self.default_grid_unit

    def _build_form(self, sections):
        return [f"{s.id}({s.end_bar - s.start_bar})" for s in sections]
