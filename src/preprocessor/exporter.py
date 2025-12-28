from __future__ import annotations

from typing import Dict, List, Tuple


def bundle_to_text(bundle: Dict) -> str:
    lines: List[str] = []

    global_meta = bundle.get("global", {})
    lines.append("[GLOBAL]")
    for key in ["BPM", "KEY", "TIME_SIG", "GRID_UNIT", "GENRE", "STYLE"]:
        value = global_meta.get(key)
        if value is not None:
            lines.append(f"{key}={value}")
    lines.append("")

    instruments = bundle.get("instruments", {})
    lines.append("[INSTRUMENTS]")
    for role in ["MELODY", "HARMONY", "BASS", "DRUMS"]:
        value = instruments.get(role)
        if value is not None:
            lines.append(f"{role}={value}")
    lines.append("")

    lines.append("[FORM]")
    form = bundle.get("form", [])
    if isinstance(form, list):
        lines.append(" > ".join(_format_form_entry(e) for e in form))
    else:
        lines.append(str(form))
    lines.append("")

    for section in bundle.get("sections", []):
        name = section.get("name", "A")
        lines.append(f"[SECTION:{name}]")
        lines.append(f"BARS={section.get('bars', 0)}")
        for key in ["BPM", "KEY", "TIME_SIG"]:
            if key in section:
                lines.append(f"{key}={section[key]}")
        lines.append("PROG=")
        lines.extend(format_prog_grid(section.get("prog_grid", [])))
        ctrl = section.get("ctrl")
        if ctrl is not None:
            lines.append(f"CTRL={ctrl}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def format_prog_grid(prog_grid: List[List[str]]) -> List[str]:
    lines: List[str] = []
    for bar in prog_grid:
        token_str = " ".join(bar)
        lines.append(f"| {token_str} |")
    return lines


def _format_form_entry(entry: Tuple[str, int]) -> str:
    if isinstance(entry, tuple) and len(entry) == 2:
        return f"{entry[0]}({entry[1]})"
    return str(entry)
