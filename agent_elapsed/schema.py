"""遥测的两级数据契约。派生量（X、n_j）一律不落盘，存了就会和源不一致。

校验的分工是刻意的：

* **步事件**只做与模型无关的边界（版本、op 枚举、单调时间、去重），其余交给离线侧；
* **任务记录**额外要求物理完整性：非有限或负数 `total_wall`、坏形状的分类概率、非字符串
  的 `model_id/host` 会污染之后每一次均值、方差与覆盖率计算，属于"这条数据没有意义"，
  不是建模选择；
* `g_pred` / `g_final` 的**完整性不在这里判**——宿主要能先落盘、再逐步补齐分类。缺列或档位
  非法的记录由 `offline.partition.partition_tasks` 隔离并计数，而不是让拟合半路抛异常。

任何违规都返回 `(None, reason)` 交给拒绝记录，**不允许**以异常打断宿主的写入路径：
宿主写坏数据是常态，一行坏数据不该让整条遥测链路停摆。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from agent_elapsed.constants import OPS
from agent_elapsed.features import FEATURE_LEVELS, G_SCHEMA_VERSION


@dataclass(frozen=True)
class StepEvent:
    task_id: str
    op: str
    t_start: float
    t_end: float
    ok: bool = True
    tool: str | None = None
    v: int = G_SCHEMA_VERSION

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    g_pred: dict
    g_pred_probs: tuple[tuple[str, float], ...]
    g_final: dict | None
    total_wall: float
    censored: bool
    model_id: str
    host: str
    estimator_wall: float
    g_schema_version: int
    v: int = G_SCHEMA_VERSION

    def env_key(self) -> str:
        return f"{self.model_id}|{self.host}"


class _Reject(Exception):
    """Internal control flow: a validation failure carrying the reject reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _key(task_id, op, t_start):
    return (task_id, op, t_start)


def _number(value, reason: str, *, non_finite: str | None = None) -> float:
    """Finite float or reject. `non_finite` keeps the historical reason for NaN/inf timestamps."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise _Reject(reason) from None
    if not math.isfinite(number):
        raise _Reject(non_finite or reason)
    return number


def _text(value, reason: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _Reject(reason)
    return value


def parse_step_row(row: dict, seen: set) -> tuple[StepEvent | None, str | None]:
    """Three model-independent checks; malformed input is a reason, never an exception."""
    try:
        if not isinstance(row, dict):
            raise _Reject("malformed_row")
        if row.get('v', G_SCHEMA_VERSION) != G_SCHEMA_VERSION:
            raise _Reject('bad_g_schema_version')
        op = row.get("op")
        if op not in OPS:
            raise _Reject("bad_op")
        task_id = _text(row.get("task_id"), "bad_task_id")
        t_start = _number(row.get("t_start"), "bad_timestamp", non_finite="non_monotonic")
        t_end = _number(row.get("t_end"), "bad_timestamp", non_finite="non_monotonic")
        if t_end <= t_start:
            raise _Reject("non_monotonic")
        tool = row.get("tool")
        if tool is not None and not isinstance(tool, str):
            raise _Reject("bad_tool")
    except _Reject as bad:
        return None, bad.reason
    k = _key(task_id, op, t_start)
    if k in seen:
        return None, "duplicate"
    seen.add(k)
    return StepEvent(task_id=task_id, op=op, t_start=t_start, t_end=t_end,
                     ok=bool(row.get("ok", True)), tool=tool), None


def parse_probs(raw) -> tuple[tuple[tuple[str, float], ...] | None, str | None]:
    """Class probabilities: known levels, finite non-negative weights.

    An unknown level or a negative/NaN weight cannot be interpreted by the model at all, so it is
    rejected on ingest rather than exploding later inside `encode_top2`.
    """
    if raw is None:
        return (), None
    try:
        if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
            raise _Reject("bad_probs")
        levels = FEATURE_LEVELS["task_class"]
        out: list[tuple[str, float]] = []
        for item in raw:
            if (isinstance(item, (str, bytes)) or not isinstance(item, (list, tuple))
                    or len(item) != 2):
                raise _Reject("bad_probs")
            level, weight = item
            if level not in levels:
                raise _Reject("bad_probs_level")
            value = _number(weight, "bad_probs_weight")
            if value < 0:
                raise _Reject("bad_probs_weight")
            out.append((str(level), value))
    except _Reject as bad:
        return None, bad.reason
    except TypeError:
        return None, "bad_probs"
    return tuple(out), None


def parse_task_row(row: dict) -> tuple[TaskRecord | None, str | None]:
    """Physical integrity only; see the module docstring for what is deliberately not checked."""
    try:
        if not isinstance(row, dict):
            raise _Reject("malformed_row")
        try:
            version = int(row.get("g_schema_version", 0))
        except (TypeError, ValueError):
            raise _Reject("bad_g_schema_version") from None
        if version != G_SCHEMA_VERSION:
            raise _Reject("bad_g_schema_version")
        if "g_pred" not in row:
            raise _Reject("missing_g_pred")
        if not isinstance(row["g_pred"], dict):
            raise _Reject("bad_g_pred")
        task_id = _text(row.get("task_id"), "bad_task_id")
        probs, reason = parse_probs(row.get("g_pred_probs", []))
        if reason:
            raise _Reject(reason)
        g_final = row.get("g_final")
        if g_final is not None and not isinstance(g_final, dict):
            raise _Reject("bad_g_final")
        total_wall = _number(row.get("total_wall"), "bad_total_wall")
        if total_wall < 0:
            raise _Reject("bad_total_wall")
        censored = row.get("censored", False)
        if not isinstance(censored, bool):
            raise _Reject("bad_censored")
        model_id = _text(row.get("model_id"), "bad_model_id")
        host = _text(row.get("host"), "bad_host")
        estimator_wall = _number(row.get("estimator_wall", 0.0), "bad_estimator_wall")
        if estimator_wall < 0:
            raise _Reject("bad_estimator_wall")
    except _Reject as bad:
        return None, bad.reason
    assert probs is not None
    return TaskRecord(
        task_id=task_id, g_pred=dict(row["g_pred"]),
        g_pred_probs=probs, g_final=dict(g_final) if g_final else None,
        total_wall=total_wall, censored=censored,
        model_id=model_id, host=host,
        estimator_wall=estimator_wall, g_schema_version=version,
    ), None
