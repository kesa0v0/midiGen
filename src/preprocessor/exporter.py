from pathlib import Path


class DatasetExporter:
    """
    Deterministic text export for conductor tokens.
    """

    ROLE_ORDER = ["MELODY", "HARMONY", "BASS", "DRUMS"]
    CTRL_ORDER = ["DYN", "DEN", "MOV", "FILL", "FEEL", "ENERGY"]

    def export(
        self,
        conductor_bundle,
        instruments,
        output_path,
        midi_path: str = ""
    ):
        global_meta = conductor_bundle["global"]
        form = conductor_bundle["form"]
        sections = conductor_bundle["sections"]

        lines = []

        # [GLOBAL]
        lines.append("[GLOBAL]")
        lines.append(f"BPM={global_meta['bpm']}")
        lines.append(f"TIME_SIG={global_meta['time_sig'][0]}/{global_meta['time_sig'][1]}")
        lines.append(f"GRID_UNIT={global_meta['grid_unit']}")
        if global_meta.get("key"):
            lines.append(f"KEY={global_meta['key']}")
        lines.append("")  # blank line

        # [INSTRUMENTS]
        lines.append("[INSTRUMENTS]")
        for role in self.ROLE_ORDER:
            val = instruments.get(role, "UNKNOWN")
            lines.append(f"{role}={val}")
        lines.append("")  # blank line

        # [FORM]
        lines.append("[FORM]")
        lines.append(" > ".join(form))

        # Sections
        for sec in sections:
            lines.append("")  # blank line
            lines.append(f"[SECTION:{sec.name}]")
            lines.append(f"BARS={sec.bars}")
            if sec.bpm is not None and sec.bpm != global_meta["bpm"]:
                lines.append(f"BPM={sec.bpm}")
            if sec.time_sig is not None and (sec.time_sig.numerator, sec.time_sig.denominator) != tuple(global_meta["time_sig"]):
                lines.append(f"TIME_SIG={sec.time_sig.numerator}/{sec.time_sig.denominator}")
            if sec.key is not None and sec.key != global_meta.get("key"):
                lines.append(f"KEY={sec.key}")

            lines.append("PROG=")
            for bar in sec.prog_grid:
                lines.append("| " + " ".join(bar) + " |")

            ctrl_parts = []
            for key in self.CTRL_ORDER:
                if key in sec.control_tokens:
                    ctrl_parts.append(f"{key}:{sec.control_tokens[key]}")
            lines.append("CTRL=" + " ".join(ctrl_parts))

        # Ensure trailing newline for UTF-8 text stability
        output_text = "\n".join(lines) + "\n"
        target_path = self._resolve_output_path(output_path, midi_path)
        with open(target_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(output_text)

    def _resolve_output_path(self, output_path, midi_path: str) -> str:
        """
        Enforce deterministic filename: <midi_stem>.tokens.txt when a directory or None is provided.
        """
        path_obj = Path(output_path) if output_path else None
        if path_obj is None or path_obj.is_dir():
            midi_stem = Path(midi_path).stem if midi_path else "output"
            target = (path_obj or Path(".")).joinpath(f"{midi_stem}.tokens.txt")
            return str(target)
        return str(path_obj)
