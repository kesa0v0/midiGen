from typing import Dict, List, Optional, Set


class PreValidator:
    """
    Structural/mechanical checks before processing.
    """

    def __init__(
        self,
        min_bars: int = 4,
        min_bpm: float = 30.0,
        max_bpm: float = 300.0,
        max_tempo_changes: int = 32,
        max_time_sig_changes: int = 16,
    ):
        self.min_bars = min_bars
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.max_tempo_changes = max_tempo_changes
        self.max_time_sig_changes = max_time_sig_changes

    def validate(self, midi) -> bool:
        # Parseable? midi object already created.
        try:
            end_time = midi.get_end_time()
        except Exception:
            return False

        # Basic length via bars heuristic (downbeats preferred)
        bars = midi.get_downbeats()
        if len(bars) < self.min_bars:
            return False

        # BPM checks
        tempos, tempi = midi.get_tempo_changes()
        if len(tempi) == 0:
            return False
        if any(t < self.min_bpm or t > self.max_bpm for t in tempi):
            return False
        if len(tempi) > self.max_tempo_changes:
            return False

        # Time signature checks
        ts_changes = midi.time_signature_changes
        if not ts_changes:
            return False
        if len(ts_changes) > self.max_time_sig_changes:
            return False

        # Note presence
        has_notes = any(inst.notes for inst in midi.instruments)
        if not has_notes:
            return False

        # All tracks empty
        if all(len(inst.notes) == 0 for inst in midi.instruments):
            return False

        return True


class PostValidator:
    """
    Grammar + quality gating after token generation.
    """

    FORBIDDEN_TOKENS: Set[str] = {"N/A", "UNK", "INVALID"}

    def __init__(
        self,
        max_sections: int = 20,
        min_sections: int = 1,
        min_section_bars: int = 2,
        max_section_bars: int = 256,
        max_repeated_chord_ratio: float = 0.9,
        min_chord_variety: int = 2,
    ):
        self.max_sections = max_sections
        self.min_sections = min_sections
        self.min_section_bars = min_section_bars
        self.max_section_bars = max_section_bars
        self.max_repeated_chord_ratio = max_repeated_chord_ratio
        self.min_chord_variety = min_chord_variety

    def validate(self, global_block: Dict, sections) -> bool:
        # Section count limits
        if not (self.min_sections <= len(sections) <= self.max_sections):
            return False

        # FORM sanity: ensure form matches section order/lengths
        form = global_block.get("form", [])
        if form and len(form) != len(sections):
            return False

        grid_unit = global_block.get("grid_unit")
        if grid_unit not in {"1/4", "1/8", "1/16"}:
            return False

        # MIDI metadata presence
        if global_block.get("midi_type") is None or global_block.get("ticks_per_beat") is None:
            return False

        for sec in sections:
            if not self._section_basic(sec):
                return False
            if not self._prog_slots(sec, grid_unit):
                return False
            if self._has_forbidden_tokens(sec):
                return False
            if not self._ctrl_valid(sec):
                return False

        if not self._music_sanity(sections):
            return False

        return True

    def _section_basic(self, sec) -> bool:
        bars = sec.bars
        if bars < self.min_section_bars or bars > self.max_section_bars:
            return False
        return True

    def _prog_slots(self, sec, grid_unit: Optional[str]) -> bool:
        slots_per_bar = getattr(sec, "slots_per_bar", None)
        if slots_per_bar is None or slots_per_bar <= 0:
            return False
        expected_slots = sec.bars * slots_per_bar
        actual_slots = sum(len(bar) for bar in sec.prog_grid)
        if expected_slots != actual_slots:
            return False
        # Optional: check BPM/TIME_SIG consistency inside section
        if sec.bpm and hasattr(sec, "local_bpm") and sec.local_bpm != sec.bpm:
            return False
        return True

    def _has_forbidden_tokens(self, sec) -> bool:
        for bar in sec.prog_grid:
            for tok in bar:
                if tok in self.FORBIDDEN_TOKENS:
                    return True
        return False

    def _ctrl_valid(self, sec) -> bool:
        ctrl = sec.control_tokens
        required_keys = {"DYN", "DEN", "MOV", "FILL", "FEEL"}
        if not required_keys.issubset(ctrl.keys()):
            return False

        dyn_ok = ctrl["DYN"] in {"LOW", "MID", "HIGH", "RISING", "FALLING"}
        den_ok = ctrl["DEN"] in {"SPARSE", "NORMAL", "DENSE"}
        mov_ok = ctrl["MOV"] in {"ASC", "DESC", "STATIC"}
        fill_ok = ctrl["FILL"] in {"YES", "NO"}
        feel_ok = ctrl["FEEL"] in {"STRAIGHT", "SWING"}
        energy_ok = True
        if "ENERGY" in ctrl:
            try:
                val = int(ctrl["ENERGY"])
                energy_ok = 1 <= val <= 5
            except Exception:
                energy_ok = False
        return dyn_ok and den_ok and mov_ok and fill_ok and feel_ok and energy_ok

    def _music_sanity(self, sections) -> bool:
        all_chords = []
        for sec in sections:
            for bar in sec.prog_grid:
                if bar:
                    all_chords.append(bar[0])

        if not all_chords:
            return False

        # Chord variety
        unique_chords = set(all_chords)
        if len(unique_chords) < self.min_chord_variety:
            return False

        # Over-repetition
        from collections import Counter

        counts = Counter(all_chords)
        most_common = counts.most_common(1)[0][1]
        if most_common / len(all_chords) > self.max_repeated_chord_ratio:
            return False

        # Section contrast: ensure at least two sections differ in chord content
        if len(sections) >= 2:
            sec_signatures = [tuple(bar[0] for bar in sec.prog_grid) for sec in sections]
            if len(set(sec_signatures)) == 1:
                return False

        return True
