"""对数步长的 Normal-Inverse-Gamma 共轭后验。参数估计做在数据稠密的步级。"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from agent_elapsed.constants import (
    OPS, PRIOR_EQUIV_N, PRIOR_MEDIAN_SECONDS, PRIOR_SIGMA, StepParams,
)


class NIG(NamedTuple):
    mu: float
    kappa: float
    alpha: float
    beta: float


def prior_for(op: str, equiv_n: int = PRIOR_EQUIV_N) -> NIG:
    alpha = equiv_n / 2.0 + 1.0
    return NIG(mu=math.log(PRIOR_MEDIAN_SECONDS[op]), kappa=float(equiv_n),
               alpha=alpha, beta=PRIOR_SIGMA ** 2 * (alpha - 1.0))


def posterior(xs, prior: NIG, weights=None) -> NIG:
    """Weighted Normal-Inverse-Gamma posterior.

    Numerical choices: the scatter is accumulated with exact summation (fsum)
    on deviations from the prior mean, so it never expands the square of a
    large number the way a Σx² − (Σx)²/n shortcut would, and the update is
    built from the mean deviation directly instead of re-centering twice. At
    realistic log-durations the plain two-pass dot product is already accurate,
    so this is robustness rather than a repair; the invariant tests pin the
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
        return prior
    shift = prior.mu                    # x̄ - μ_0 == mean deviation from the shift
    d = xs - shift
    sw = math.fsum(w * d)
    sw2 = math.fsum(w * d * d)
    mean_dev = sw / n                   # = x̄ - μ_0
    scatter = max(sw2 - sw * mean_dev, 0.0)   # = Σ w (x - x̄)², without cancellation
    kappa_n = prior.kappa + n
    mu_n = (prior.kappa * prior.mu + n * (shift + mean_dev)) / kappa_n
    beta_n = (prior.beta + 0.5 * scatter
              + prior.kappa * n * mean_dev ** 2 / (2.0 * kappa_n))
    return NIG(mu=mu_n, kappa=kappa_n, alpha=prior.alpha + n / 2.0, beta=beta_n)


def sigma_hat(p: NIG) -> float:
    if p.alpha <= 1.0:
        raise ValueError("后验形状不足，无法给出有限的 σ 估计")
    return math.sqrt(p.beta / (p.alpha - 1.0))


def fit_step_distributions(steps, equiv_n: int = PRIOR_EQUIV_N, weights=None) -> dict[str, StepParams]:
    """删失任务的步级观测照常计入：已发生的每一步时长本身是完整观测。"""
    buckets: dict[str, list[float]] = {op: [] for op in OPS}
    bucket_weights = {op: [] for op in OPS}
    steps = list(steps)
    if weights is not None and len(weights) != len(steps):
        raise ValueError('步级权重必须与样本对齐')
    for i, ev in enumerate(steps):
        if ev.duration > 0:
            buckets[ev.op].append(math.log(ev.duration))
            bucket_weights[ev.op].append(1.0 if weights is None else weights[i])
    out = {}
    for op in OPS:
        p = posterior(np.array(buckets[op]), prior_for(op, equiv_n), bucket_weights[op])
        out[op] = StepParams(mu=p.mu, sigma=sigma_hat(p), step_obs=len(buckets[op]))
    return out
