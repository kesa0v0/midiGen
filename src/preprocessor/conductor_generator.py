from collections import Counter
from typing import Dict, List, Optional, Tuple

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
        chord_detect_grid_unit = self._chord_detect_grid_unit(analysis, global_ts)
        chord_grid_unit, chord_grid_mode, chord_grid_stats = self._chord_grid_unit(
            analysis,
            global_ts,
            sections,
            prog_extractor,
            midi=midi,
            detect_grid_unit=chord_detect_grid_unit,
        )
        midi_type = analysis.get("midi_type")
        ticks_per_beat = analysis.get("ticks_per_beat")
        channel_programs = analysis.get("channel_programs")

        conductor_sections = []

        for sec in sections:
            local_bpm = sec.local_bpm or global_bpm
            local_ts = sec.local_time_sig or TimeSignature(*global_ts)
            local_key = sec.local_key or global_key

            detect_slots_per_bar = self._slots_per_bar(local_ts, chord_detect_grid_unit)
            export_slots_per_bar = self._slots_per_bar(local_ts, chord_grid_unit)
            prog, prog_ext, used_slots = prog_extractor.extract(
                midi,
                sec,
                analysis,
                detect_slots_per_bar,
                export_slots_per_bar=export_slots_per_bar,
            )
            ctrl = ctrl_extractor.extract(midi, sec, analysis)

            conductor_sections.append(
                ConductorSection(
                    name=sec.instance_id,
                    bars=sec.end_bar - sec.start_bar,
                    bpm=local_bpm,
                    time_sig=local_ts,
                    key=local_key,
                    prog_grid=prog,
                    prog_ext_grid=prog_ext,
                    control_tokens=ctrl,
                    slots_per_bar=used_slots,
                )
            )

        self._align_repeated_sections(conductor_sections, sections)
        form = self._build_form(sections)
        self._apply_hooks(conductor_sections, sections, analysis, form)

        return {
            "global": {
                "bpm": global_bpm,
                "time_sig": global_ts,
                "key": global_key,
                "grid_unit": grid_unit,
                "chord_grid_unit": chord_grid_unit,
                "chord_detect_grid_unit": chord_detect_grid_unit,
                "chord_export_grid_mode": chord_grid_mode,
                "chord_export_grid_selected": chord_grid_unit,
                "chord_export_grid_stats": chord_grid_stats,
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

    def _chord_grid_unit(
        self,
        analysis,
        global_ts,
        sections,
        prog_extractor,
        midi=None,
        detect_grid_unit: str = "1/4",
    ):
        if "chord_grid_unit" in analysis:
            return analysis["chord_grid_unit"], "FIXED", "source=chord_grid_unit"
        if "harmonic_grid_unit" in analysis:
            return analysis["harmonic_grid_unit"], "FIXED", "source=harmonic_grid_unit"
        if analysis.get("adaptive_chord_grid", True) and midi is not None:
            grid, stats = self._adaptive_chord_grid_unit(
                analysis,
                global_ts,
                sections,
                prog_extractor,
                midi,
                detect_grid_unit,
            )
            return grid, "ADAPTIVE", stats
        return self._default_chord_grid_unit(global_ts), "FIXED", "source=default"

    def _chord_detect_grid_unit(self, analysis, global_ts):
        if "chord_detect_grid_unit" in analysis:
            return analysis["chord_detect_grid_unit"]
        if "harmonic_detect_grid_unit" in analysis:
            return analysis["harmonic_detect_grid_unit"]
        _, den = global_ts
        if den >= 8:
            return "1/8"
        return "1/4"

    def _default_chord_grid_unit(self, global_ts):
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

    def _adaptive_chord_grid_unit(
        self,
        analysis,
        global_ts,
        sections,
        prog_extractor,
        midi,
        detect_grid_unit: str,
    ) -> Tuple[str, str]:
        if not sections:
            return self._default_chord_grid_unit(global_ts), "stats=empty_sections"

        run_lengths_bars: List[float] = []
        loss_1_2_groups: List[float] = []
        loss_1_1_groups: List[float] = []

        analysis_tmp = dict(analysis)
        analysis_tmp["absolute_chords"] = True
        analysis_tmp["chord_detail_mode"] = "split"

        for sec in sections:
            local_ts = sec.local_time_sig or TimeSignature(*global_ts)
            detect_slots_per_bar = self._slots_per_bar(local_ts, detect_grid_unit)
            if detect_slots_per_bar <= 0:
                continue
            prog_grid, _, _ = prog_extractor.extract(
                midi,
                sec,
                analysis_tmp,
                detect_slots_per_bar,
                export_slots_per_bar=detect_slots_per_bar,
            )
            if not prog_grid:
                continue
            expanded = self._expand_prog_grid(prog_grid)
            run_lengths_bars.extend(self._run_lengths_bars(expanded, detect_slots_per_bar))

            slots_half = self._slots_per_bar(local_ts, "1/2")
            loss_1_2_groups.extend(
                self._merge_loss(expanded, detect_slots_per_bar, slots_half)
            )

            slots_whole = self._slots_per_bar(local_ts, "1/1")
            loss_1_1_groups.extend(
                self._merge_loss(expanded, detect_slots_per_bar, slots_whole)
            )

        if not run_lengths_bars:
            return self._default_chord_grid_unit(global_ts), "stats=empty_runs"

        median_change_bars = float(np.median(run_lengths_bars))
        loss_1_2 = float(np.mean(loss_1_2_groups)) if loss_1_2_groups else None
        loss_1_1 = float(np.mean(loss_1_1_groups)) if loss_1_1_groups else None

        loss_1_2_threshold = 0.35
        loss_1_1_threshold = 0.12
        loss_delta_threshold = 0.05
        min_change_for_whole = 1.5
        loss_delta = None
        if loss_1_1 is not None and loss_1_2 is not None:
            loss_delta = loss_1_1 - loss_1_2

        if (
            median_change_bars <= 0.25
            and loss_1_2 is not None
            and loss_1_2 >= loss_1_2_threshold
        ):
            selected = "1/4"
        elif loss_delta is not None and loss_delta >= loss_delta_threshold:
            selected = "1/2"
        elif (
            median_change_bars >= min_change_for_whole
            and loss_1_1 is not None
            and loss_1_1 <= loss_1_1_threshold
        ):
            selected = "1/1"
        else:
            selected = "1/2"

        stats = self._grid_stats_string(
            median_change_bars,
            loss_1_2,
            loss_1_1,
            len(run_lengths_bars),
            len(loss_1_2_groups),
            len(loss_1_1_groups),
            loss_delta,
        )
        return selected, stats

    def _grid_stats_string(
        self,
        median_change_bars: float,
        loss_1_2: Optional[float],
        loss_1_1: Optional[float],
        run_samples: int,
        groups_1_2: int,
        groups_1_1: int,
        loss_delta: Optional[float],
    ) -> str:
        return (
            f"median_change_bars={self._fmt_optional(median_change_bars)}"
            f",loss_1_2={self._fmt_optional(loss_1_2)}"
            f",loss_1_1={self._fmt_optional(loss_1_1)}"
            f",loss_delta={self._fmt_optional(loss_delta)}"
            f",run_samples={run_samples}"
            f",groups_1_2={groups_1_2}"
            f",groups_1_1={groups_1_1}"
        )

    def _fmt_optional(self, value: Optional[float]) -> str:
        if value is None:
            return "NA"
        return f"{value:.3f}"

    def _run_lengths_bars(self, tokens: List[str], slots_per_bar: int) -> List[float]:
        runs: List[float] = []
        prev = None
        run = 0
        for tok in tokens:
            if tok == "N.C.":
                if prev is not None:
                    runs.append(run / slots_per_bar)
                prev = None
                run = 0
                continue
            if prev is None:
                prev = tok
                run = 1
                continue
            if tok == prev:
                run += 1
            else:
                runs.append(run / slots_per_bar)
                prev = tok
                run = 1
        if prev is not None:
            runs.append(run / slots_per_bar)
        return runs

    def _merge_loss(
        self,
        tokens: List[str],
        detect_slots_per_bar: int,
        target_slots_per_bar: int,
    ) -> List[float]:
        if target_slots_per_bar <= 0:
            return []
        if detect_slots_per_bar % target_slots_per_bar != 0:
            return []
        group = detect_slots_per_bar // target_slots_per_bar
        if group <= 1:
            return []

        losses: List[float] = []
        for i in range(0, len(tokens), group):
            group_tokens = [tok for tok in tokens[i : i + group] if tok != "N.C."]
            if not group_tokens:
                continue
            counts = Counter(group_tokens)
            dominant = max(counts.values())
            loss = 1.0 - (dominant / len(group_tokens))
            losses.append(loss)
        return losses


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

    def _align_repeated_sections(self, conductor_sections, sections, similarity_threshold: float = 0.75):
        groups = {}
        for idx, sec in enumerate(sections):
            groups.setdefault(sec.type_id, []).append(idx)

        for indices in groups.values():
            if len(indices) < 2:
                continue
            entries = []
            for idx in indices:
                csec = conductor_sections[idx]
                expanded = self._expand_prog_grid(csec.prog_grid)
                if not expanded:
                    continue
                entries.append((idx, expanded, csec))
            if len(entries) < 2:
                continue

            length_groups = {}
            for idx, expanded, csec in entries:
                length_groups.setdefault(len(expanded), []).append((idx, expanded, csec))

            for same_len in length_groups.values():
                if len(same_len) < 2:
                    continue

                sig_counts = {}
                for idx, expanded, csec in same_len:
                    sig = tuple(expanded)
                    sig_counts.setdefault(sig, []).append((idx, expanded, csec))

                best_sig, best_entries = max(sig_counts.items(), key=lambda item: len(item[1]))
                if len(best_entries) == 1:
                    best_idx, _, base_csec = max(
                        same_len, key=lambda item: self._prog_confidence(item[1])
                    )
                    best_sig = tuple(self._expand_prog_grid(base_csec.prog_grid))
                else:
                    best_idx = best_entries[0][0]
                    base_csec = conductor_sections[best_idx]

                for idx, expanded, csec in same_len:
                    if idx == best_idx:
                        continue
                    similarity = self._token_similarity(best_sig, expanded)
                    if similarity >= similarity_threshold:
                        csec.prog_grid = [list(row) for row in base_csec.prog_grid]
                        if base_csec.prog_ext_grid is not None:
                            csec.prog_ext_grid = [list(row) for row in base_csec.prog_ext_grid]

    def _expand_prog_grid(self, prog_grid):
        expanded = []
        prev = None
        for bar in prog_grid:
            for tok in bar:
                if tok == "-" and prev is not None:
                    expanded.append(prev)
                else:
                    expanded.append(tok)
                    prev = tok
        return expanded

    def _token_similarity(self, tokens_a, tokens_b) -> float:
        if not tokens_a or not tokens_b or len(tokens_a) != len(tokens_b):
            return 0.0
        matches = sum(1 for a, b in zip(tokens_a, tokens_b) if a == b)
        return matches / len(tokens_a)

    def _prog_confidence(self, tokens) -> float:
        if not tokens:
            return 0.0
        non_nc = sum(1 for t in tokens if t != "N.C.")
        return non_nc / len(tokens)

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

