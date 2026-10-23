"""对数步长的 Normal-Inverse-Gamma 共轭后验。参数估计做在数据稠密的步级。

这里有一个不显然、但会决定产品能否自校正的选择：**预测用哪一个 σ**。

完整的 NIG 后验把"先验位置与数据位置不一致"也算进 σ 的后验：

    β_n = β0 + ½·Σw(x-x̄)² + κ0·n·(x̄-μ0)²/(2·(κ0+n))

第三项（`location_gap`）随 n 增大趋于常数 κ0·Δ²/2，而第二项按 n 增长。当真实离散度本来就小
（构建步骤永远 3 秒那种）而冷启动先验中位数猜错了一个数量级时，第三项会长期压过第二项，
σ̂ 退化成 ~1/√n 的伪离散度：区间被系统性放大 → 覆盖率冲到 1.0 → KS 拒绝 → **第一次重拟合
永远过不了验收门**，偏偏那正是新宿主最需要它的时候。

把位置与尺度分开估（先估 μ，再在给定 μ 下估 σ）是标准的经验贝叶斯拆分：位置先验只是冷启动
猜测，数据够了就该让数据说话；尺度仍受先验收缩（n 小的时候）。所以预测用的 σ 取 `scatter`
口径（不含 location_gap），而 location_gap 作为诊断量输出（`step_diagnostics`），
让运维看见"先验中位数错得离谱，该重新 init"，而不是让它悄悄变成一条更宽的区间。
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from agent_elapsed.constants import (
    LOCATION_PRIOR_N, OPS, PRIOR_EQUIV_N, PRIOR_MEDIAN_SECONDS, PRIOR_SIGMA, StepParams,
)


class NIG(NamedTuple):
    mu: float
    kappa: float
    alpha: float
    beta: float


class PosteriorSummary(NamedTuple):
    """Full NIG posterior plus the pieces the point estimate has to choose between."""
    full: NIG
    scatter: float          # Σw(x-x̄)²
    n: float                # Σw
    mean_dev: float         # x̄ - μ0
    location_gap: float     # κ0·n·Δ²/(2·(κ0+n))，只属于 μ 的不确定性

    def sigma_scatter(self, prior: NIG) -> float:
        """Dispersion with the location gap excluded — what predictions must use."""
        alpha = prior.alpha + self.n / 2.0
        if alpha <= 1.0:
            raise ValueError("后验形状不足，无法给出有限的 σ 估计")
        return math.sqrt(max(prior.beta + 0.5 * self.scatter, 0.0) / (alpha - 1.0))


def prior_for(op: str, equiv_n: int = PRIOR_EQUIV_N,
              location_n: float = LOCATION_PRIOR_N) -> NIG:
    alpha = equiv_n / 2.0 + 1.0
    return NIG(mu=math.log(PRIOR_MEDIAN_SECONDS[op]), kappa=float(location_n),
               alpha=alpha, beta=PRIOR_SIGMA ** 2 * (alpha - 1.0))


def summarise(xs, prior: NIG, weights=None) -> PosteriorSummary:
    """Weighted NIG posterior; scatter accumulated exactly (fsum) around the prior mean.

    Centring on a fixed shift avoids expanding the square of a large number the way a
    Σx² − (Σx)²/n shortcut would. At realistic log-durations the plain two-pass dot product is
    already accurate, so this is robustness rather than a repair; the invariant tests pin the
    result to a 60-digit decimal reference.
    """
    xs = np.asarray(xs, dtype=float)
    w = np.ones(xs.size) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != xs.shape or np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError('步级权重必须有限、非负且与样本对齐')
    if not np.all(np.isfinite(xs)):
        raise ValueError('对数步长必须有限')
    n = math.fsum(w)
    if n == 0:
        return PosteriorSummary(prior, 0.0, 0.0, 0.0, 0.0)
    shift = prior.mu                    # x̄ - μ_0 == mean deviation from the shift
    d = xs - shift
    sw = math.fsum(w * d)
    sw2 = math.fsum(w * d * d)
    mean_dev = sw / n                   # = x̄ - μ_0
    scatter = max(sw2 - sw * mean_dev, 0.0)   # = Σ w (x - x̄)², without cancellation
    kappa_n = prior.kappa + n
    mu_n = (prior.kappa * prior.mu + n * (shift + mean_dev)) / kappa_n
    location_gap = prior.kappa * n * mean_dev ** 2 / (2.0 * kappa_n)
    full = NIG(mu=mu_n, kappa=kappa_n, alpha=prior.alpha + n / 2.0,
               beta=prior.beta + 0.5 * scatter + location_gap)
    return PosteriorSummary(full, scatter, n, mean_dev, location_gap)


def posterior(xs, prior: NIG, weights=None) -> NIG:
    """Full NIG posterior (location gap included) — the textbook object."""
    return summarise(xs, prior, weights).full


def sigma_hat(p: NIG) -> float:
    if p.alpha <= 1.0:
        raise ValueError("后验形状不足，无法给出有限的 σ 估计")
    return math.sqrt(p.beta / (p.alpha - 1.0))


def _buckets(steps, weights):
    steps = list(steps)
    if weights is not None and len(weights) != len(steps):
        raise ValueError('步级权重必须与样本对齐')
    logs: dict[str, list[float]] = {op: [] for op in OPS}
    bucket_weights: dict[str, list[float]] = {op: [] for op in OPS}
    for i, ev in enumerate(steps):
        if ev.duration > 0:
            logs[ev.op].append(math.log(ev.duration))
            bucket_weights[ev.op].append(1.0 if weights is None else weights[i])
    return logs, bucket_weights


def fit_step_distributions(steps, equiv_n: int = PRIOR_EQUIV_N, weights=None,
                          dispersion: str = "scatter") -> dict[str, StepParams]:
    """Fit one lognormal per operation; `dispersion="scatter"` excludes the location gap.

    Censored tasks are fine here: a step that happened is a complete observation, even if the
    task around it was cancelled.
    """
    if dispersion not in ("scatter", "full"):
        raise ValueError('dispersion 只能是 "scatter"（预测用）或 "full"（完整后验）')
    logs, bucket_weights = _buckets(steps, weights)
    out = {}
    for op in OPS:
        prior = prior_for(op, equiv_n)
        summary = summarise(np.array(logs[op]), prior, bucket_weights[op])
        sigma = (summary.sigma_scatter(prior) if dispersion == "scatter"
                 else sigma_hat(summary.full))
        out[op] = StepParams(mu=summary.full.mu, sigma=sigma, step_obs=len(logs[op]))
    return out


def step_diagnostics(steps, equiv_n: int = PRIOR_EQUIV_N, weights=None) -> dict[str, dict]:
    """Per-operation evidence for operators: is the cold-start median still credible?

    `location_gap_share` is the fraction of the full posterior's β that comes from the prior
    location disagreeing with the data. Large values mean the cold-start median is wrong and
    that folding the gap into σ would inflate every interval — which is why the point estimate
    uses the scatter-only dispersion instead.
    """
    logs, bucket_weights = _buckets(steps, weights)
    out = {}
    for op in OPS:
        prior = prior_for(op, equiv_n)
        summary = summarise(np.array(logs[op]), prior, bucket_weights[op])
        beta_total = prior.beta + 0.5 * summary.scatter + summary.location_gap
        out[op] = {
            "step_obs": len(logs[op]),
            "median_seconds": math.exp(summary.full.mu) if summary.n else None,
            "prior_median_seconds": PRIOR_MEDIAN_SECONDS[op],
            "mean_log_gap": round(summary.mean_dev, 4),
            "sigma_scatter": summary.sigma_scatter(prior) if summary.n else None,
            "sigma_full": sigma_hat(summary.full),
            "location_gap_share": round(summary.location_gap / beta_total, 4)
            if beta_total > 0 else 0.0,
        }
    return out
