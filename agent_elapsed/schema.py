"""遥测的两级数据契约。派生量（X、n_j）一律不落盘，存了就会和源不一致。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from agent_elapsed.constants import OPS
from agent_elapsed.features import G_SCHEMA_VERSION


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


def _key(task_id, op, t_start):
    return (task_id, op, t_start)


def parse_step_row(row: dict, seen: set) -> tuple[StepEvent | None, str | None]:
    if row.get('v', G_SCHEMA_VERSION) != G_SCHEMA_VERSION:
        return None, 'bad_g_schema_version'
    op = row.get("op")
    if op not in OPS:
        return None, "bad_op"
    t_start = float(row["t_start"])
    t_end = float(row["t_end"])
    if not math.isfinite(t_start) or not math.isfinite(t_end) or t_end <= t_start:
        return None, "non_monotonic"
    k = _key(row["task_id"], op, t_start)
    if k in seen:
        return None, "duplicate"
    seen.add(k)
    return StepEvent(task_id=row["task_id"], op=op, t_start=t_start, t_end=t_end,
                     ok=bool(row.get("ok", True)), tool=row.get("tool")), None


def parse_task_row(row: dict) -> tuple[TaskRecord | None, str | None]:
    version = int(row.get("g_schema_version", 0))
    if version != G_SCHEMA_VERSION:
        return None, "bad_g_schema_version"
    if "g_pred" not in row:
        return None, "missing_g_pred"
    return TaskRecord(
        task_id=row["task_id"], g_pred=dict(row["g_pred"]),
        g_pred_probs=tuple((str(k), float(v)) for k, v in row.get("g_pred_probs", [])),
        g_final=dict(row["g_final"]) if row.get("g_final") else None,
        total_wall=float(row["total_wall"]), censored=bool(row.get("censored", False)),
        model_id=row["model_id"], host=row["host"],
        estimator_wall=float(row.get("estimator_wall", 0.0)),
        g_schema_version=version,
    ), None
