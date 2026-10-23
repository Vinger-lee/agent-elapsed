"""Shared display formatting; contains no estimation or offline imports."""
from __future__ import annotations

import math


def _int_band(lo: float, hi: float) -> tuple[int, int]:
    lo_i, hi_i = int(round(lo)), int(round(hi))
    if hi_i <= lo_i:
        hi_i = max(int(math.ceil(hi)), lo_i + 1)
    return lo_i, hi_i


def format_band(p50: float, p90: float) -> str:
    if max(p50, p90) < 60:
        lo, hi = _int_band(p50, p90)
        return f"大约 {lo}–{hi} 秒"
    lo, hi = _int_band(p50 / 60, p90 / 60)
    return f"大约 {lo}–{hi} 分钟"


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"已用 {max(0, int(seconds))} 秒"
    return f"已用 {int(seconds / 60)} 分钟"
