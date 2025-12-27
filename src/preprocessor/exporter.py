from pathlib import Path


class DatasetExporter:
    """
    Deterministic text export for conductor tokens.
    """

    ROLE_ORDER = [
        "INSTRUMENT", 
        "MELODY", "HARMONY", "BASS", "DRUMS", 
        "STRINGS", "BRASS", "WOODWINDS", "PERCUSSION"
    ]
    CTRL_ORDER = ["DYN", "DEN", "MOV", "FILL", "FEEL", "LEAP", "SPACING", "ENERGY"]

    def export(
        self,
        conductor_bundle,
        instruments,
        output_path,
        midi_path: str = "",
        genre: str = "UNKNOWN",
        style: str = "UNKNOWN",
        artist: str = "UNKNOWN",
        inst_type: str = "UNKNOWN"
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
        if global_meta.get("chord_grid_unit"):
            lines.append(f"CHORD_GRID_UNIT={global_meta['chord_grid_unit']}")
        if global_meta.get("chord_detect_grid_unit"):
            lines.append(f"CHORD_DETECT_GRID_UNIT={global_meta['chord_detect_grid_unit']}")
        if global_meta.get("chord_export_grid_mode"):
            lines.append(f"CHORD_EXPORT_GRID_MODE={global_meta['chord_export_grid_mode']}")
        if global_meta.get("chord_export_grid_selected"):
            lines.append(f"CHORD_EXPORT_GRID_SELECTED={global_meta['chord_export_grid_selected']}")
        if global_meta.get("chord_export_grid_stats"):
            lines.append(f"CHORD_EXPORT_GRID_STATS={global_meta['chord_export_grid_stats']}")
        if global_meta.get("key"):
            lines.append(f"KEY={global_meta['key']}")
        lines.append(f"GENRE={genre}")
        lines.append(f"STYLE={style}")
        lines.append(f"ARTIST={artist}")
        lines.append(f"INST_TYPE={inst_type}")
        lines.append("")  # blank line

        # [INSTRUMENTS]
        lines.append("[INSTRUMENTS]")
        for role in self.ROLE_ORDER:
            if role in instruments:
                val = instruments[role]
                if val != "NONE":
                    lines.append(f"{role}={val}")
        
        # Fallback: Print any other keys in instruments not in ROLE_ORDER (sorted)
        for key in sorted(instruments.keys()):
            if key not in self.ROLE_ORDER:
                val = instruments[key]
                if val != "NONE":
                    lines.append(f"{key}={val}")
                    
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
            lines.append(f"HOOK={sec.hook or 'NO'}")
            lines.append(f"HOOK_REPEAT={sec.hook_repeat or 'VARIATION'}")
            if sec.hook_role:
                lines.append(f"HOOK_ROLE={sec.hook_role}")
            if sec.hook_range:
                lines.append(f"HOOK_RANGE={sec.hook_range}")
            if sec.hook_rhythm:
                lines.append(f"HOOK_RHYTHM={sec.hook_rhythm}")

            lines.append("PROG=")
            for bar in sec.prog_grid:
                lines.append("| " + " ".join(bar) + " |")

            if sec.prog_ext_grid:
                lines.append("PROG_EXT=")
                for bar in sec.prog_ext_grid:
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
