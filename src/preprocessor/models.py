from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TimeSignature:
    numerator: int
    denominator: int


@dataclass
class Section:
    id: str
    start_bar: int
    end_bar: int
    local_bpm: Optional[int] = None
    local_time_sig: Optional[TimeSignature] = None
    local_key: Optional[str] = None
    role: Optional[str] = None


@dataclass
class ConductorSection:
    name: str
    bars: int
    bpm: Optional[int]
    time_sig: Optional[TimeSignature]
    key: Optional[str]
    prog_grid: List[List[str]]
    control_tokens: Dict[str, str]
    slots_per_bar: int = 0
    hook: Optional[str] = None
    hook_repeat: Optional[str] = None
    hook_role: Optional[str] = None
    hook_range: Optional[str] = None
    hook_rhythm: Optional[str] = None
