"""覆盖率分数本身就是漂移探测器：换模型、换机器、装新工具都会让常数过期。"""
from __future__ import annotations

import math

import numpy as np

from offline.calibrate import interval_loss
from offline.gate import samples_for_task
from offline.partition import partition_tasks


def per_task_pinball(table, tasks, m: int = 2000, seed_base: int = 7) -> list[float]:
    """Scores of the tasks the model can read; unreadable rows are dropped, not scored as zero.

    Censored tasks are kept: this is a monitoring series, and the caller decides its scope
    (`refit`/`report` already pass complete tasks only).
    """
    out = []
    for i, task in enumerate(partition_tasks(tasks, table).usable):
        s = samples_for_task(table, task, m=m, seed=seed_base + i)
        out.append(interval_loss(float(task.total_wall), s))
    return out


def ewma_line(scores, alpha: float = 0.1) -> list[dict]:
    """滚动 EWMA 均值与标准差；upper = 均值 + 2σ，就是 v1 的那条 2σ 线。"""
    out = []
    mean = var = 0.0
    for i, s in enumerate(scores):
        if i == 0:
            mean, var = float(s), 0.0
        else:
            diff = float(s) - mean
            mean = mean + alpha * diff
            var = (1 - alpha) * (var + alpha * diff * diff)
        std = math.sqrt(max(var, 0.0))
        out.append({"i": i, "score": float(s), "ewma": mean, "std": std,
                    "upper": mean + 2.0 * std})
    return out


def detect(scores, baseline_n: int = 30, k: float = 2.0) -> dict:
    """最近 100 项的快速 EWMA：log1p 压低长尾，连续两次越过基线 2σ 才报警。

    基线固定在已知稳定段；不把正在检测的数据吸入基线而掩盖永久漂移。
    """
    if len(scores) <= baseline_n:
        return {"alert": False, "index": None, "z": 0.0}
    values = np.log1p(np.asarray(scores, dtype=float))
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('pinball 分数必须有限且非负')
    base = values[:baseline_n]
    mu, sd = float(base.mean()), float(base.std(ddof=1) or 0.0)
    if sd <= 0:
        sd = float(np.mean(np.abs(base - mu)) or 1e-9)
    streak, z = 0, 0.0
    for i in range(baseline_n, len(values)):
        line = ewma_line(values[max(0, i - 99):i + 1], alpha=0.5)
        z = (line[-1]['ewma'] - mu) / sd
        streak = streak + 1 if z > k else 0
        if streak >= 2:
            return {"alert": True, "index": i, "z": z}
    return {"alert": False, "index": None, "z": float(z)}


def discount_weights(n: int, half_life_tasks: int = 120) -> np.ndarray:
    """指数衰减到均值 1：折扣只改变相对权重，不该悄悄改变惩罚强度。"""
    if n <= 0:
        return np.zeros(0)
    idx = np.arange(n, dtype=float)
    w = np.exp(-math.log(2.0) / float(half_life_tasks) * (float(n - 1) - idx))
    return w / float(w.mean())
