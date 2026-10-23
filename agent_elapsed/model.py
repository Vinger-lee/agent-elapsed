"""复合泊松求和 T = Σ_j Σ_{i<=n_j} X_{ji}。分位数靠蒙特卡洛，不允许正态近似。

矩的闭式解：步长对数正态的有效对数方差 s² = σ²·max(φ,1)，
有效对数均值同时下移 (s²-σ²)/2，于是

    E[X]   = exp(μ + σ²/2)          —— 与 φ 无关
    E[X²]  = E[X]² · exp(s²)        —— φ 只放大二阶矩

φ 因此是**纯离散度**参数：它把相对方差抬起来而不偷偷抬高均值。
旧实现只放大 σ 而不补偿 μ，φ>1 会同时抬高均值和中位数之外的尺度，
令 gate 的离散度估计与它想修的问题不是同一个量（见 docs/math-upgrade.md）。
"""
from __future__ import annotations

import math

import numpy as np

from agent_elapsed.constants import ETA_CLIP, OPS, ConstTable
from agent_elapsed.features import encode_top2

# 正态 95% 双侧分位数：分位数区间的秩估计用它换算成标准误。
QUANTILE_CI_Z = 1.959963984540054
# 一次预估至少要两条路径，否则"分位数区间"退化成一个点，任何统计量都无从谈起。
MIN_PATHS = 2
# 单次预估最多展开的步数。上限不是性能调优，而是安全阀：λ 一旦被拟合成天文数字，
# 采样会去分配上百 GB 的数组并把宿主进程 OOM 掉——宁可抛一个可读的错误让宿主降级。
MAX_EVENTS = 20_000_000


def _rates(table: ConstTable, vec) -> dict[str, float]:
    rates = {}
    for op in OPS:
        eta = table.intercept[op] + math.fsum(c * x for c, x in zip(table.coef[op], vec))
        rates[op] = math.exp(min(max(eta, -ETA_CLIP), ETA_CLIP))
    return rates


def _components(table: ConstTable, g: dict, probs):
    return [(_rates(table, vec), weight) for vec, weight in encode_top2(g, probs)]


def lambdas(table: ConstTable, g: dict, probs) -> dict[str, float]:
    """Mean rates for expectation/contributions, not rates of a mixture Poisson."""
    components = _components(table, g, probs)
    return {op: math.fsum(weight * rates[op] for rates, weight in components) for op in OPS}


def step_log_moments(table: ConstTable, op: str) -> tuple[float, float]:
    """(log E[X], log E[X²]) under the mean-preserving dispersion of φ.

    Both quantities are read off the effective parameters only, so callers
    cannot accidentally mix the φ-widened σ with the un-shifted μ.
    """
    mu_eff, sigma_eff = table.log_params(op)
    log_mean = mu_eff + sigma_eff ** 2 / 2.0
    log_second = 2.0 * log_mean + sigma_eff ** 2
    return log_mean, log_second


def step_moments(table: ConstTable, op: str) -> tuple[float, float]:
    """(E[X], E[X²]) with one exp per moment instead of two.  Raises on non-finite."""
    log_mean, log_second = step_log_moments(table, op)
    try:
        mean = math.exp(log_mean)
        second = math.exp(log_second)
    except OverflowError as exc:                      # pragma: no cover - guarded by schema
        raise ValueError(f"{op} 的步长矩溢出：{exc}") from exc
    if not (math.isfinite(mean) and math.isfinite(second)):
        raise ValueError(f"{op} 的步长矩必须是有限数")
    return mean, second


def _moments(table: ConstTable) -> dict[str, tuple[float, float]]:
    return {op: step_moments(table, op) for op in OPS}


def expected_by_op(table: ConstTable, lam: dict[str, float],
                   moments=None) -> dict[str, float]:
    moments = moments or _moments(table)
    return {op: lam[op] * moments[op][0] for op in OPS}


def expected_total(table: ConstTable, lam: dict[str, float], moments=None) -> float:
    return math.fsum(expected_by_op(table, lam, moments).values())


def variance_total(table: ConstTable, lam: dict[str, float], moments=None) -> float:
    """Conditional compound-Poisson variance is lambda*E[X^2], not lambda*Var(X)."""
    moments = moments or _moments(table)
    return math.fsum(lam[op] * moments[op][1] for op in OPS)


def cv_squared(table: ConstTable, g: dict, probs = None, moments=None) -> float:
    """Var(T)/E[T]² (φ-independent denominator, φ-monotone numerator)."""
    moments = moments or _moments(table)
    lam = lambdas(table, g, probs)
    mean = expected_total(table, lam, moments)
    if mean <= 0:
        return 0.0
    return variance_total(table, lam, moments) / mean ** 2


def task_moments(table: ConstTable, g: dict, probs) -> dict[str, float]:
    components = _components(table, g, probs)
    moments = _moments(table)                 # 步长矩与类别无关，只算一次
    means = [expected_total(table, rates, moments) for rates, _ in components]
    mean = math.fsum(weight * value for (_, weight), value in zip(components, means))
    within = math.fsum(weight * variance_total(table, rates, moments)
                       for rates, weight in components)
    between = math.fsum(weight * (value - mean) ** 2
                        for (_, weight), value in zip(components, means))
    return {"expected": mean, "within_class_variance": within,
            "between_class_variance": between, "variance": within + between}


def quantile_interval(samples: np.ndarray, tau: float,
                      z: float = QUANTILE_CI_Z) -> tuple[float, float]:
    """Distribution-free interval for the population τ-quantile.

    Inverts the exact binomial rank test: the count below the population
    quantile is Binomial(m, tau), so ranks floor(m·tau ∓ z·sqrt(m·tau·(1-tau)))
    bracket it.  No density estimate is needed, which matters for the atom at
    zero where a kernel/SE formula would be wrong.
    """
    if samples.size == 0:
        raise ValueError("样本为空，无法给出分位数区间")
    if not 0.0 < tau < 1.0:
        raise ValueError("tau 必须在 (0,1) 内")
    ordered = np.sort(samples)
    m = ordered.size
    centre = m * tau
    half = z * math.sqrt(m * tau * (1.0 - tau))
    lo = int(min(max(math.floor(centre - half), 0), m - 1))
    hi = int(min(max(math.ceil(centre + half), 0), m - 1))
    return float(ordered[lo]), float(ordered[hi])


def quantile_se(samples: np.ndarray, tau: float, z: float = QUANTILE_CI_Z) -> float:
    """Monte Carlo standard error implied by the rank interval."""
    lo, hi = quantile_interval(samples, tau, z)
    return (hi - lo) / (2.0 * z)


def _check_paths(m: int) -> int:
    if isinstance(m, bool) or not isinstance(m, int) or m < MIN_PATHS:
        raise ValueError(f"m 必须是 ≥{MIN_PATHS} 的整数路径数，收到 {m!r}")
    return m


def _expected_events(lam, m: int) -> float:
    """How many step samples this draw would materialise, before allocating anything."""
    total = 0.0
    for op in OPS:
        value = lam[op]
        if isinstance(value, np.ndarray):
            total += float(np.sum(value))          # 每路径一个率时，和已经是总数
        else:
            total += float(value) * m
    return total


def sample_totals(table: ConstTable, lam: dict[str, float], m: int, seed: int) -> np.ndarray:
    return _sample_totals(table, lam, _check_paths(m), np.random.default_rng(seed))


def _sample_totals(table: ConstTable, lam, m: int, rng) -> np.ndarray:
    events = _expected_events(lam, m)
    if not math.isfinite(events) or events > MAX_EVENTS:
        per_path = math.fsum(float(np.sum(lam[op])) if isinstance(lam[op], np.ndarray)
                             else float(lam[op]) for op in OPS)
        raise ValueError(
            f"这次预估需要展开约 {events:,.0f} 步，超过单次上限 {MAX_EVENTS:,}"
            f"（当前每路径期望 {per_path:,.1f} 步 × {m} 条路径）。"
            "通常是常数表的计数参数 λ 被拟合成异常值，或 m 过大；"
            "请检查常数表与分类概率，而不是把宿主进程拖进 OOM。")
    total = np.zeros(m)
    for op in OPS:                      # OPS 顺序固定，采样消耗顺序才可复现
        counts = rng.poisson(lam[op], size=m)
        n = int(counts.sum())
        if n == 0:
            continue
        mu_eff, sigma_eff = table.log_params(op)
        x = np.exp(rng.normal(mu_eff, sigma_eff, size=n))
        total += np.bincount(np.repeat(np.arange(m), counts), weights=x, minlength=m)
    return total


def sample_task(table: ConstTable, g: dict, probs, m: int, seed: int) -> np.ndarray:
    _check_paths(m)
    components = _components(table, g, probs)
    if len(components) == 1:
        return sample_totals(table, components[0][0], m, seed)
    rng = np.random.default_rng(seed)
    # One latent class per task path, shared across all operation types.
    chosen = rng.choice(len(components), size=m, p=[weight for _, weight in components])
    rates = {op: np.array([lam[op] for lam, _ in components])[chosen] for op in OPS}
    return _sample_totals(table, rates, m, rng)


def predict(table: ConstTable, g: dict, probs, m: int = 5000, seed: int = 20261022) -> dict:
    _check_paths(m)
    probs = list(probs or [])
    lam = lambdas(table, g, probs)
    moments = _moments(table)
    means = expected_by_op(table, lam, moments)
    expected = math.fsum(means.values())
    samples = sample_task(table, g, probs, m, seed)
    shares = {op: (means[op] / expected if expected > 0 else 0.0) for op in OPS}
    p50, p90 = float(np.quantile(samples, 0.5)), float(np.quantile(samples, 0.9))
    p50_lo, p50_hi = quantile_interval(samples, 0.5)
    p90_lo, p90_hi = quantile_interval(samples, 0.9)
    return {
        "p50": p50,
        "p90": p90,
        "p50_se": (p50_hi - p50_lo) / (2.0 * QUANTILE_CI_Z),
        "p90_se": (p90_hi - p90_lo) / (2.0 * QUANTILE_CI_Z),
        "p50_ci": (p50_lo, p50_hi),
        "p90_ci": (p90_lo, p90_hi),
        "expected": expected,
        "contributions": shares,
        "lambdas": lam,
        "samples": samples,
        "m": m,
        "seed": seed,
    }
