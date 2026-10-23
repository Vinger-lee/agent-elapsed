"""g 的分类档位 → 设计矩阵列。LLM 只出档位；数字只能在下游算出来，不能由它给。"""
from __future__ import annotations

import math

G_SCHEMA_VERSION = 1

FEATURE_LEVELS: dict[str, tuple[str, ...]] = {
    "task_class": ("question", "locate", "patch", "refactor",
                   "scaffold", "debug", "research"),
    "touched_files": ("1", "2-3", "4-10", ">10"),
    "verification_loops": ("0", "1", "2", ">2"),
    "needs_network": ("false", "true"),
    "plan_steps": ("0", "1-3", "4-8", ">8"),
}

FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{name}={level}" for name, levels in FEATURE_LEVELS.items() for level in levels
)
N_FEATURES = len(FEATURE_COLUMNS)
_INDEX = {c: i for i, c in enumerate(FEATURE_COLUMNS)}


class FeatureError(ValueError):
    pass


def _level(name: str, value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def encode(g: dict) -> list[float]:
    missing = [k for k in FEATURE_LEVELS if k not in g]
    if missing:
        raise FeatureError(f"缺少特征：{missing}")
    vec = [0.0] * N_FEATURES
    for name, levels in FEATURE_LEVELS.items():
        level = _level(name, g[name])
        if level not in levels:
            raise FeatureError(f"{name} 取值非法：{g[name]!r}，可选 {levels}")
        vec[_INDEX[f"{name}={level}"]] = 1.0
    return vec


def encode_top2(g: dict, probs) -> list[tuple[list[float], float]]:
    """只混合 task_class —— 分类噪声的主要来源，其余特征视为确定。"""
    if not probs:
        return [(encode(g), 1.0)]
    selected = [(level, float(weight)) for level, weight in list(probs)[:2]]
    if any(not math.isfinite(weight) or weight < 0 for _, weight in selected):
        raise FeatureError('分类概率必须是有限的非负数')
    scale = max((weight for _, weight in selected), default=0.0)
    if scale <= 0:
        return [(encode(g), 1.0)]
    # Scale before summing so finite weights cannot overflow the normalizer.
    scaled = [(level, weight / scale) for level, weight in selected]
    total = math.fsum(weight for _, weight in scaled)
    out = []
    for level, weight in scaled:
        out.append((encode({**g, "task_class": level}), weight / total))
    return out
