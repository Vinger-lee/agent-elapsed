"""拟合结果不直接上线：先打赢线上版，再原子替换。"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agent_elapsed.constants import (
    MC_PATHS, MIN_FIT_TASKS, MIN_TOTAL_TASKS, OPS, PHI_MAX, PHI_MIN, ConstTable,
    load_tables, prior_table,
)
from agent_elapsed.model import lambdas, step_moments, task_moments
from offline.calibrate import calibration_report, interval_loss, samples_for_task
from offline.fit_counts import fit_counts
from offline.fit_step import fit_step_distributions
from offline.partition import partition_tasks

SEED_BASE = 20261022


def holdout_size(n_total: int) -> int:
    return max(20, math.ceil(0.2 * n_total))


def temporal_split(tasks):
    """时间序切分。随机切会因为漂移高估精度（spec §6.2）。"""
    n = len(tasks)
    cut = max(0, n - holdout_size(n))
    return list(tasks[:cut]), list(tasks[cut:])


def gate_eligible(n_total: int, n_fit: int) -> bool:
    return n_total >= MIN_TOTAL_TASKS and n_fit >= MIN_FIT_TASKS


def _predict(table: ConstTable, task, m: int, seed: int) -> np.ndarray:
    return samples_for_task(table, task, m=m, seed=seed)


def estimate_phi(table: ConstTable, tasks, m: int = MC_PATHS, seed_base: int = SEED_BASE) -> float:
    """Solve the residual-dispersion of φ ∈ [1, 25]; not a free-form ratio.

    φ only scales the log-variance (E[X] is φ-free, see ConstTable.log_params),
    so the model's mean relative variance is a monotone function of φ:

        mean_t[Var(T_t)/E[T_t]²] = Σ_j B_j · exp(φ · σ_j²)

    with B_j = mean_t[λ_t,j · E[X_j]² / E[T_t]²] and E[T_t] φ-independent. The
    fixed point is found by bisection on that strictly increasing function,
    which makes the returned φ an exact match of the holdout's mean squared
    relative residual instead of a heuristic multiplier (docs/math-upgrade.md).
    m/seed_base remain for call compatibility; no sampling is needed here.
    """
    moments = {op: step_moments(table, op) for op in OPS}
    variance2 = {op: table.step[op].sigma ** 2 for op in OPS}
    basis = {op: 0.0 for op in OPS}
    between_relative = 0.0
    target = 0.0
    n = 0
    for task in partition_tasks(tasks, table, complete_only=True).usable:
        lam = lambdas(table, task.g_pred, list(task.g_pred_probs))
        moments_task = task_moments(table, task.g_pred, list(task.g_pred_probs))
        mean = moments_task["expected"]
        if mean <= 0:
            continue
        scale = 1.0 / mean ** 2
        for op in OPS:
            basis[op] += lam[op] * moments[op][0] ** 2 * scale
        # 类别间方差与 φ 无关（φ 只放大步长对数方差），必须作为常数项留着，
        # 否则混合分布的离散度会被算成“还差得远”，φ 一路夹到上限。
        between_relative += moments_task["between_class_variance"] * scale
        target += ((float(task.total_wall) - mean) / mean) ** 2
        n += 1
    if n == 0:
        return PHI_MIN
    basis = {op: value / n for op, value in basis.items()}
    between_relative /= n
    target /= n

    def excess(phi: float) -> float:
        return (between_relative
                + math.fsum(basis[op] * _exp_overflow_inf(phi * variance2[op])
                            for op in OPS) - target)

    if excess(PHI_MIN) >= 0.0:            # 模型已经够散，不再夸大
        return PHI_MIN
    if excess(PHI_MAX) <= 0.0:            # 连上限都不够：夹住，别假装修好
        return PHI_MAX
    lo, hi = PHI_MIN, PHI_MAX
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if excess(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def _exp_overflow_inf(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:                 # 单调函数只需符号，inf 是正确答案
        return math.inf


def build_candidate(fit_steps, fit_tasks, env: str = "default", weights=None) -> ConstTable:
    base = prior_table(env)
    fit_tasks = list(fit_tasks)
    if weights is not None and len(weights) != len(fit_tasks):
        raise ValueError('任务权重必须与样本对齐')
    selected = [i for i,t in enumerate(fit_tasks) if t.g_schema_version == base.g_schema_version]
    tasks = [fit_tasks[i] for i in selected]
    if weights is not None:
        weights = [weights[i] for i in selected]
    fit_ids = {t.task_id for t in tasks}
    fit_steps = [s for s in fit_steps if s.task_id in fit_ids]
    step_weights = None
    if weights is not None:
        by_id = {t.task_id:float(w) for t,w in zip(tasks,weights)}
        step_weights = [by_id[s.task_id] for s in fit_steps]
    step = fit_step_distributions(fit_steps, weights=step_weights)
    counts = fit_counts(fit_steps, tasks, ridge=1.0, weights=weights)
    trial = ConstTable(
        env=env, g_schema_version=base.g_schema_version, fit_id="candidate",
        trained_through_ts=0.0, phi=1.0, task_count=sum(not t.censored for t in tasks), ks_passed=False,
        intercept={o: counts[o][0] for o in OPS},
        coef={o: counts[o][1] for o in OPS}, step=step)
    ts = max([e.t_end for e in fit_steps], default=0.0)
    phi = estimate_phi(trial, tasks, m=1500, seed_base=SEED_BASE)
    payload = json.dumps({"step": {o: (step[o].mu, step[o].sigma, step[o].step_obs) for o in OPS},
                          "counts": counts, "phi": phi, "env": env,
                          "version": base.g_schema_version}, sort_keys=True)
    fit_id = f"fit-{int(ts)}-{hashlib.sha1(payload.encode()).hexdigest()[:7]}"
    return ConstTable(
        env=env, g_schema_version=base.g_schema_version, fit_id=fit_id,
        trained_through_ts=float(ts), phi=phi, task_count=sum(not t.censored for t in tasks),
        ks_passed=False, intercept={o: counts[o][0] for o in OPS},
        coef={o: counts[o][1] for o in OPS}, step=step)


@dataclass
class GateResult:
    accepted: bool
    reason: str
    details: dict = field(default_factory=dict)


def evaluate(online: ConstTable, candidate: ConstTable, holdout_tasks,
             require_strict: bool = False, min_effect: float = 0.0) -> GateResult:
    """同一组完整验证任务配对比较：至少 20 验证、60 拟合、合计 80。
    candidate.task_count 在上线前指完整拟合任务数；CLI 另对整批检查。

    min_effect 是“算作进步”的最小配对均值改进（默认 0，即保持原判据）；
    配对差的标准误一并输出，好让拒绝/接受的理由带着不确定度，而不是
    把一次蒙特卡洛抖动当成模型进步。
    """
    partition = partition_tasks(holdout_tasks, candidate, complete_only=True)
    holdout_tasks = partition.usable
    n = len(holdout_tasks)
    if n < 20 or not gate_eligible(candidate.task_count + n, candidate.task_count):
        return GateResult(False, "insufficient-tasks",
                          {"n_holdout": n, "unusable_holdout": partition.summary()})
    report = calibration_report(candidate, holdout_tasks, m=2000, seed_base=SEED_BASE)
    seeds = [SEED_BASE + i for i in range(n)]
    differences, cand_pin, onl_pin = [], [], []
    for i, task in enumerate(holdout_tasks):
        y = float(task.total_wall)
        # 同一任务、同一种子，两个模型各跑一次 MC —— 这才叫配对比较
        cs = _predict(candidate, task, 2000, seeds[i])
        os_ = _predict(online, task, 2000, seeds[i])
        cand_pin.append(interval_loss(y, cs))
        onl_pin.append(interval_loss(y, os_))
        differences.append(cand_pin[-1] - onl_pin[-1])
    cand_mean, onl_mean = float(np.mean(cand_pin)), float(np.mean(onl_pin))
    diff = np.asarray(differences, dtype=float)
    diff_mean = float(diff.mean())
    diff_se = float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else float("inf")
    z = diff_mean / diff_se if diff_se > 0 else 0.0
    details = {"n_holdout": n, "candidate_pinball": cand_mean, "online_pinball": onl_mean,
               "paired_diff": diff_mean, "paired_diff_se": diff_se, "paired_z": z,
               "unusable_holdout": partition.summary(),
               "per_task_seeds": seeds, "calibration_pass": report["pass"], "report": report}
    if not report["pass"]:
        return GateResult(False, "calibration-failed", details)
    if require_strict and not diff_mean < -min_effect:
        return GateResult(False, "not strictly better", details)
    if diff_mean > min_effect:
        return GateResult(False, "worse than online", details)
    return GateResult(True, "accepted", details)


def rollback(path) -> bool:
    path = Path(path)
    prev = path.with_suffix(".prev.json")
    if not prev.exists():
        return False
    load_tables(prev)
    os.replace(prev, path)
    return True
