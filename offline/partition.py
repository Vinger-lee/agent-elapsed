"""把"记录在库里、但模型读不了"的任务隔离出来并计数。

契约层（`agent_elapsed.schema`）刻意不要求宿主一次就把 `g_pred` 填全：真实集成里任务先落盘、
分类稍后补。代价是离线侧每次拿任务列表都可能遇到缺列或非法档位。这里提供唯一的隔离入口——
隔离要**算得清**（返回原因与条数，进 report 和 CLI 输出），不能靠 try/except 静默吞掉：
被吞掉的往往不是随机丢失，而是某一类任务，静默丢会系统性偏移拟合。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent_elapsed.features import FeatureError, encode_top2


@dataclass(frozen=True)
class Partition:
    usable: list = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)

    def reasons(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for _, reason in self.skipped:
            out[reason] = out.get(reason, 0) + 1
        return out

    def summary(self) -> dict:
        return {"usable": len(self.usable), "skipped": self.n_skipped,
                "reasons": self.reasons()}


def encode_reason(g_pred, probs) -> str | None:
    """Why the model cannot read this prediction, or None if it can."""
    try:
        encode_top2(g_pred, list(probs or []))
    except FeatureError as exc:
        return f"features: {exc}"
    except (TypeError, ValueError, AttributeError):
        return "features: malformed"
    return None


def partition_tasks(tasks, table=None, *, complete_only: bool = False) -> Partition:
    """Split tasks into usable ones and `(task_id, reason)` pairs for the rest.

    `complete_only` additionally drops censored tasks, because their wall time is only a lower
    bound: they may inform step lengths but never counts, coverage or acceptance.
    """
    usable, skipped = [], []
    for task in tasks:
        task_id = getattr(task, "task_id", "<unknown>")
        if complete_only and getattr(task, "censored", False):
            skipped.append((task_id, "censored"))
            continue
        if table is not None and getattr(task, "g_schema_version", None) != table.g_schema_version:
            skipped.append((task_id, "g_schema_version"))
            continue
        prob_list = list(getattr(task, "g_pred_probs", None) or [])
        reason = encode_reason(getattr(task, "g_pred", None), prob_list)
        if reason:
            skipped.append((task_id, reason))
            continue
        usable.append(task)
    return Partition(usable, skipped)
