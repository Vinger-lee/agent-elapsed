"""Plain text widget for the host's existing layout; never paints the terminal."""
from __future__ import annotations

import re
import unicodedata

from agent_elapsed.hooks import WidgetState

_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _clean(text: str) -> str:
    text = _CSI.sub("", _OSC.sub("", text))
    return " ".join("".join(
        " " if char.isspace() else char
        for char in text if char.isspace() or unicodedata.category(char) not in ("Cc", "Cf")
    ).split())


def _cells(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1


def _fit(text: str, width: int) -> str:
    if sum(_cells(char) for char in text) <= width:
        return text
    fitted, used = [], 0
    for char in text:
        cells = _cells(char)
        if used + cells > width - 1:
            break
        fitted.append(char)
        used += cells
    return "".join(fitted) + "…"


def render_tui(state: WidgetState, *, width: int = 80) -> str:
    if isinstance(width, bool) or not isinstance(width, int) or width < 8:
        raise ValueError("width must be an integer of at least 8 terminal cells")
    lines = []
    for task in state.tasks:
        data = task.to_dict()
        detail = f"{data['band'] or '任务进行中'} · {data['elapsed_label']}"
        if data["calibrating"]:
            detail += " · 校准中"
        lines.extend((_fit("● " + _clean(task.title), width), _fit("  " + detail, width)))
        if data["over_estimate"]:
            lines.append(_fit("  已超过预估区间", width))
    return "\n".join(lines)
