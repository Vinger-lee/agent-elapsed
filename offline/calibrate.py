"""区间预测器只能用覆盖率评价；MAPE 分不清"自信的错"和"诚实的宽"（spec §9.5）。"""
from __future__ import annotations

import math

import numpy as np

from agent_elapsed.constants import MC_PATHS, OPS, ConstTable
from agent_elapsed.model import sample_task
from offline.partition import partition_tasks

TAUS = (0.5, 0.8, 0.9)


def samples_for_task(table: ConstTable, task, m: int = MC_PATHS, seed: int = 7) -> np.ndarray:
    return sample_task(table, task.g_pred, list(task.g_pred_probs), m, seed)


def pit_of(actual: float, samples: np.ndarray) -> float:
    """Empirical CDF for diagnostics; atoms require randomization for a uniform PIT."""
    return float(np.mean(samples <= actual))


def randomized_pit(actual: float, samples: np.ndarray, u: float) -> float:
    """Randomized rank among m predictions plus the held-out observation.

    Independent U(0, 1) tie-breaking makes this uniform for exchangeable
    actual/predictive draws, including the compound-Poisson atom at zero.
    """
    less = int(np.count_nonzero(samples < actual))
    equal = int(np.count_nonzero(samples == actual))
    return (less + u * (equal + 1)) / (samples.size + 1)


def coverage_tolerance(tau: float, n: int) -> float:
    """两个标准误。定死的 ±0.05 在 n=20 时约等于 1σ，会频繁误杀合格模型。"""
    return 2.0 * math.sqrt(tau * (1.0 - tau) / float(n))


def ks_stat_uniform(pits) -> float:
    u = np.sort(np.asarray(pits, dtype=float))
    n = u.size
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def ks_pvalue_uniform(d: float, n: int) -> float:
    if n <= 0:
        return 0.0
    if d <= 0:
        return 1.0
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * float(d)
    total = 0.0
    for k in range(1, 101):
        sign = -1.0 if (k - 1) % 2 else 1.0
        term = sign * math.exp(-2.0 * (k * lam) ** 2)
        total += term
        if abs(term) < 1e-14:
            break
    return float(min(max(2.0 * total, 0.0), 1.0))


def pinball(y: float, q: float, tau: float) -> float:
    return ((1.0 if y <= q else 0.0) - tau) * (q - y)


def interval_loss(y: float, samples: np.ndarray,
                  taus: tuple[float, float] = (0.5, 0.9)) -> float:
    """Mean pinball loss of the displayed quantiles; single source of truth.

    gate.evaluate and drift.per_task_pinball must score a task the same way or
    their numbers stop being comparable, so the formula lives here only.
    """
    return math.fsum(pinball(y, float(np.quantile(samples, tau)), tau)
                     for tau in taus) / len(taus)


def calibration_report(table: ConstTable, tasks, m: int = MC_PATHS,
                       seed_base: int = 7) -> dict:
    partition = partition_tasks(tasks, table, complete_only=True)
    tasks = partition.usable
    unusable = partition.summary()
    n = len(tasks)
    if n == 0:
        return {"n": 0, "unusable": unusable, "taus": {},
                "ks": {"d": 1.0, "p": 0.0, "pass": False},
                "pinball": {"p50": float("inf"), "p90": float("inf"), "mean": float("inf")},
                "pass": False}
    actuals = np.array([t.total_wall for t in tasks])
    qs = {}
    pins = {}
    pits = []
    pit_rng = np.random.default_rng([seed_base, 0x504954])
    for i, task in enumerate(tasks):
        samples = samples_for_task(table, task, m=m, seed=seed_base + i)
        for tau in TAUS:
            qs.setdefault(tau, []).append(float(np.quantile(samples, tau)))
        pits.append(randomized_pit(float(task.total_wall), samples, float(pit_rng.random())))
        pins.setdefault("p50", []).append(pinball(float(task.total_wall), qs[0.5][i], 0.5))
        pins.setdefault("p90", []).append(pinball(float(task.total_wall), qs[0.9][i], 0.9))
    taus_out = {}
    ok = True
    for tau in TAUS:
        cov = float(np.mean(actuals <= np.array(qs[tau])))
        below = float(np.mean(actuals < np.array(qs[tau])))
        tol = coverage_tolerance(tau, n)
        # A discrete quantile brackets tau: P(Y<q) <= tau <= P(Y<=q).
        passed = below <= tau + tol and cov >= tau - tol
        taus_out[f"{tau}"] = {"coverage": cov, "coverage_below": below,
                               "tol": tol, "pass": passed}
        ok = ok and passed
    d = ks_stat_uniform(pits)
    p = ks_pvalue_uniform(d, n)
    ks = {"d": d, "p": p, "pass": p > 0.05}
    pin_mean = float(np.mean(pins["p50"] + pins["p90"]))
    return {"n": n, "unusable": unusable, "taus": taus_out, "ks": ks,
            "pinball": {"p50": float(np.mean(pins["p50"])),
                        "p90": float(np.mean(pins["p90"])), "mean": pin_mean},
            "pass": bool(ok and ks["pass"])}
