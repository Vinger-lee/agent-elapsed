"""Shared display formatting; contains no estimation or offline imports."""
from __future__ import annotations

import math


def _int_band(lo: float, hi: float) -> tuple[int, int]:
    lo_i, hi_i = int(round(lo)), int(round(hi))
    if hi_i <= lo_i:
        hi_i = max(int(math.ceil(hi)), lo_i + 1)
    return lo_i, hi_i


def format_band(p50: float, p90: float) -> str:
    """Legacy two-sided band. Kept for callers that formatted it themselves; see below."""
    if max(p50, p90) < 60:
        lo, hi = _int_band(p50, p90)
        return f"大约 {lo}–{hi} 秒"
    lo, hi = _int_band(p50 / 60, p90 / 60)
    return f"大约 {lo}–{hi} 分钟"


def format_expectation(p50: float, p90: float) -> str:
    """What the用户-facing widget says, and why it is worded this way.

    P50–P90 is a 40% band for a continuous calibrated distribution, not a 90% one
    (`docs/accuracy-audit.md`). Printing it as "大约 3–9 分钟" reads like "it will land in
    there", which overstates the interval by more than a factor of two in probability. So the
    widget states the two numbers it actually knows: the median, and the 90% upper bound.
    """
    if max(p50, p90) < 60:
        lo, hi = _int_band(p50, p90)
        return f"通常 {lo} 秒 · 九成在 {hi} 秒内"
    lo, hi = _int_band(p50 / 60, p90 / 60)
    return f"通常 {lo} 分钟 · 九成在 {hi} 分钟内"


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"已用 {max(0, int(seconds))} 秒"
    return f"已用 {int(seconds / 60)} 分钟"
