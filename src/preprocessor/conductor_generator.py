import numpy as np

from models import ConductorSection, TimeSignature


class ConductorTokenGenerator:
    def __init__(self, default_grid_unit: str = "1/4"):
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
        # grid_unit??臾댁“嫄?1/4濡?怨좎젙 (遺꾩꽍 寃곌낵 臾댁떆)
        grid_unit = self._grid_unit(analysis, global_ts)
        chord_grid_unit = self._chord_grid_unit(analysis, global_ts)
        midi_type = analysis.get("midi_type")
        ticks_per_beat = analysis.get("ticks_per_beat")
        channel_programs = analysis.get("channel_programs")

        conductor_sections = []

        for sec in sections:
            local_bpm = sec.local_bpm or global_bpm
            local_ts = sec.local_time_sig or TimeSignature(*global_ts)
            local_key = sec.local_key or global_key

            slots_per_bar = self._slots_per_bar(local_ts, chord_grid_unit)
            prog = prog_extractor.extract(midi, sec, analysis, slots_per_bar)
            ctrl = ctrl_extractor.extract(midi, sec, analysis)

            conductor_sections.append(
                ConductorSection(
                    name=sec.instance_id,
                    bars=sec.end_bar - sec.start_bar,
                    bpm=local_bpm,
                    time_sig=local_ts,
                    key=local_key,
                    prog_grid=prog,
                    control_tokens=ctrl,
                    slots_per_bar=slots_per_bar,
                )
            )

        form = self._build_form(sections)
        self._apply_hooks(conductor_sections, sections, analysis, form)

        return {
            "global": {
                "bpm": global_bpm,
                "time_sig": global_ts,
                "key": global_key,
                "grid_unit": grid_unit,
                "chord_grid_unit": chord_grid_unit,
                "midi_type": midi_type,
                "ticks_per_beat": ticks_per_beat,
                "channel_programs": channel_programs,
                "form": self._build_form(sections),
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

    def _chord_grid_unit(self, analysis, global_ts):
        if "chord_grid_unit" in analysis:
            return analysis["chord_grid_unit"]
        if "harmonic_grid_unit" in analysis:
            return analysis["harmonic_grid_unit"]
        _, den = global_ts
        if den >= 8:
            return "1/8"
        return "1/4"

    def _build_form(self, sections):
        return [f"{s.instance_id}({s.end_bar - s.start_bar})" for s in sections]

    def _slots_per_bar(self, time_sig: TimeSignature, grid_unit: str) -> int:
        num = time_sig.numerator
        den = time_sig.denominator
        try:
            _, g_den = grid_unit.split("/")
            g_den = int(g_den)
        except Exception:
            g_den = 4
        slots = int(num * (g_den / den))
        return max(1, slots)

    def _apply_hooks(self, conductor_sections, sections, analysis, form):
        # HOOK/REPEAT ?먯젙 媛쒖꽑: MAIN_THEME??臾댁“嫄?HOOK, 諛섎났 肄붾뱶 吏꾪뻾? IDENTICAL
        prog_hashes = {}
        for idx, (csec, sec) in enumerate(zip(conductor_sections, sections)):
            # Use role metadata instead of ID string
            is_main_theme = (sec.role == "MAIN_THEME")
            
            # prog_grid瑜??댁떆濡?蹂??(tuple濡?蹂????hash)
            prog_tuple = tuple(tuple(row) for row in csec.prog_grid)
            
            # MAIN_THEME??湲곕낯?곸쑝濡?IDENTICAL (Chorus logic replacement)
            if is_main_theme:
                repeat_type = "IDENTICAL"
            else:
                repeat_type = "VARIATION"
            
            hook = "NO"
            # 諛섎났 ?먯깋: ?숈씪 ID???댁쟾 ?뱀뀡怨?肄붾뱶 吏꾪뻾???꾩쟾??媛숈쑝硫?IDENTICAL
            found_identical = False
            is_first_occurrence = True
            
            for prev_idx in range(idx):
                prev_sec = sections[prev_idx]
                if sec.type_id == prev_sec.type_id:
                    is_first_occurrence = False
                    prev_prog_tuple = tuple(tuple(row) for row in conductor_sections[prev_idx].prog_grid)
                    if prog_tuple == prev_prog_tuple:
                        repeat_type = "IDENTICAL"
                        found_identical = True
                        break
                    # MAIN_THEME 諛섎났 異쒗쁽?댁?留?肄붾뱶媛 ?ㅻⅤ硫?VARIATION
                    if is_main_theme:
                        repeat_type = "VARIATION"
            
            # HOOK ?먯젙: MAIN_THEME??臾댁“嫄?YES, 諛섎났(IDENTICAL)?대㈃ YES
            if is_main_theme or repeat_type == "IDENTICAL":
                hook = "YES"
            else:
                hook = "NO"
            
            csec.hook = hook
            csec.hook_repeat = repeat_type
            
            # HOOK_ROLE ?ㅼ젙
            if hook == "YES":
                csec.hook_role = "MELODY"
            elif idx == 0: 
                # First section often acts as Intro/Motif
                csec.hook_role = "MOTIF"
            else:
                csec.hook_role = None
            
            csec.hook_range = self._hook_range(csec)
            csec.hook_rhythm = self._hook_rhythm(csec)

    def _base_label(self, name: str) -> str:
        return name.split("_")[0] if "_" in name else name

    def _hook_presence(self, idx: int, sections, bar_slice, base_counts) -> bool:
        cond = 0
        sec = sections[idx]
        base = self._base_label(sec.type_id)
        if base_counts.get(base, 0) > 1:
            cond += 1
        if sec.end_bar - sec.start_bar >= 8:
            cond += 1
        if self._density_boost(idx, sections, bar_slice):
            cond += 1
        if self._contour_repetition(bar_slice):
            cond += 1
        return cond >= 2

    def _density_boost(self, idx: int, sections, bar_slice) -> bool:
        if not bar_slice:
            return False
        mean_den = np.mean([b.get("note_count", 0) for b in bar_slice])
        if idx > 0:
            prev_slice = sections[idx - 1]
        else:
            prev_slice = None
        if idx + 1 < len(sections):
            next_slice = sections[idx + 1]
        else:
            next_slice = None
        neighbors = []
        bars = []
        # This function receives only bar_slice; use global analysis bars via slices
        # Simplify: compare to average density over all bars
        return mean_den >= 2.0

    def _contour_repetition(self, bar_slice) -> bool:
        cents = [b.get("pitch_centroid") for b in bar_slice if b.get("pitch_centroid") is not None]
        if len(cents) < 3:
            return False
        unique = len(set(int(c // 1) for c in cents))
        return unique <= len(cents) * 0.6

    # _hook_repeat?????댁긽 ?ъ슜?섏? ?딆쓬 (濡쒖쭅 ?듯빀)
    # def _hook_repeat(self, csec, all_secs, base_counts) -> str:
    #     pass

    def _hook_range(self, csec) -> str:
        spacing = csec.control_tokens.get("SPACING")
        if spacing == "WIDE":
            return "WIDE"
        return "NARROW"

    def _hook_rhythm(self, csec) -> str:
        feel = csec.control_tokens.get("FEEL")
        if feel == "SWING":
            return "SYNCOPATED"
        return "STRAIGHT"

