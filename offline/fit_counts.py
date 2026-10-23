"""步数回归：截距不惩罚 + 21 列 ridge 系数，Newton/IRLS 求解带回溯。"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from agent_elapsed.constants import ETA_CLIP, OPS, PRIOR_LAMBDA0
from agent_elapsed.features import G_SCHEMA_VERSION, N_FEATURES, encode


def design_rows(g_pred_list) -> np.ndarray:
    return np.array([encode(g) for g in g_pred_list], dtype=float).reshape(-1, N_FEATURES)


def task_op_counts(steps) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for ev in steps:
        per = out.setdefault(ev.task_id, {op: 0 for op in OPS})
        per[ev.op] += 1
    return out


def _objective(A, y, wts, ridge: float, p) -> float:
    eta = np.clip(A @ p, -ETA_CLIP, ETA_CLIP)
    penalty = 0.5 * float(ridge) * float(np.dot(p[1:], p[1:]))   # 截距不受惩罚
    return float(np.sum(wts * (-y * eta + np.exp(eta))) + penalty)


def _penalised_gradient(A, y, wts, ridge: float, p) -> np.ndarray:
    eta = np.clip(A @ p, -ETA_CLIP, ETA_CLIP)
    grad = A.T @ (wts * (np.exp(eta) - y))
    grad[1:] += float(ridge) * p[1:]
    return grad


def _solve_spd(hess, grad) -> np.ndarray:
    """Cholesky first: the IRLS Hessian is symmetric positive definite.

    Cholesky both halves the cost and refuses indefinite input instead of
    silently returning a rubbish step the way an LU solve can.
    """
    try:
        factor = np.linalg.cholesky(hess)
        return cho_solve(factor, grad)
    except np.linalg.LinAlgError:
        pass
    jitter = np.eye(hess.shape[0]) * max(1e-12, 1e-12 * float(np.abs(np.diag(hess)).max()))
    try:
        return cho_solve(np.linalg.cholesky(hess + jitter), grad)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(hess, grad, rcond=None)[0]


def cho_solve(factor, rhs) -> np.ndarray:
    """Solve (L Lᵀ) x = rhs from a Cholesky factor without forming the inverse."""
    y = np.linalg.solve(factor, rhs)
    return np.linalg.solve(factor.T, y)


def fit_ridge_poisson(X, y, ridge: float = 1.0, weights=None,
                      iters: int = 60, tol: float = 1e-9, gtol: float = 1e-8):
    """Weighted ridge Poisson regression with a Newton/IRLS solve and backtracking.

    Improvements over the naive loop: the intercept starts at the *weighted*
    mean rate (the unweighted mean is wrong under task discounting), the
    penalty is applied as a diagonal (it is one), Cholesky replaces LU, and
    convergence is tested on the gradient so a flat objective cannot stop the
    iteration early with a half-solved direction.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    wts = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if wts.shape != (n,) or np.any(wts < 0) or not np.all(np.isfinite(wts)):
        raise ValueError('任务权重必须有限、非负且与样本对齐')
    if wts.mean() > 0:
        wts = wts / wts.mean()          # 归一化，避免折扣整体缩放改变惩罚的相对强度
    ridge = float(ridge)
    p = np.zeros(k + 1)
    mean_w = float(np.sum(wts))
    rate = float(np.dot(wts, y) / mean_w) if mean_w > 0 else 0.0
    p[0] = math.log(max(rate, 0.05))
    for _ in range(iters):
        grad = _penalised_gradient(A, y, wts, ridge, p)
        if float(np.max(np.abs(grad))) <= gtol:
            break
        eta = np.clip(A @ p, -ETA_CLIP, ETA_CLIP)
        mu = np.exp(eta)
        hess = A.T @ (A * (wts * mu)[:, None])
        hess[np.diag_indices_from(hess)] += ridge
        hess[0, 0] -= ridge             # 截距不受惩罚（spec §3.5）
        delta = _solve_spd(hess, grad)
        step, base = 1.0, _objective(A, y, wts, ridge, p)
        while step > 1e-7:
            cand = p - step * delta
            if _objective(A, y, wts, ridge, cand) <= base + 1e-12 * max(1.0, abs(base)):
                break
            step /= 2.0
        else:
            break                        # 回溯失败：已到局部最优
        p_new = p - step * delta
        improved = float(np.max(np.abs(p_new - p)))
        p = p_new
        if improved < tol:
            break
    return float(p[0]), p[1:]


def fit_counts(steps, tasks, ridge: float = 1.0, weights=None):
    fallback = {op: (math.log(PRIOR_LAMBDA0[op]),
                     tuple(0.0 for _ in range(N_FEATURES))) for op in OPS}
    counts = task_op_counts(steps)
    usable = [t for t in tasks
              if not t.censored and t.g_schema_version == G_SCHEMA_VERSION]
    if not usable:
        return fallback
    X = design_rows([t.g_pred for t in usable])   # 只用 g_pred；读 g_final 即泄漏
    wts = None
    if weights is not None:
        all_ids = [t.task_id for t in tasks]
        by_id = {tid: i for i, tid in enumerate(all_ids)}
        wts = np.array([float(weights[by_id[t.task_id]]) for t in usable])
    out = {}
    for op in OPS:
        y = np.array([float(counts.get(t.task_id, {}).get(op, 0)) for t in usable])
        b, w = fit_ridge_poisson(X, y, ridge=ridge, weights=wts)
        out[op] = (b, tuple(float(x) for x in w))
    return out
