"""按已知真值生成遥测。它的价值不是占位，而是给出真实日志永远给不出的东西：真值。"""
from __future__ import annotations

import math

import numpy as np

from agent_elapsed.constants import OPS, PRIOR_LAMBDA0, PRIOR_MEDIAN_SECONDS, PRIOR_SIGMA
from agent_elapsed.features import FEATURE_LEVELS, FEATURE_COLUMNS

CENSORED_RATE = 0.1
CLASSIFY_ACCURACY = 0.75
MODEL_IDS = ("m-a", "m-b")
HOSTS = ("h-a",)

# 全部与 spec §3.5 的先验不同：中位数更大、更散，且有非零系数。
HIDDEN_TRUTH = {
    "step": {
        "llm": (math.log(9.0), 0.90),
        "tool": (math.log(2.2), 0.72),
        "build": (math.log(210.0), 1.10),
        "net": (math.log(6.0), 0.88),
    },
    "intercept": {
        "llm": math.log(10.0), "tool": math.log(15.0),
        "build": math.log(2.0), "net": math.log(2.5),
    },
    "coef": {
        "task_class=refactor": {"llm": 0.30, "tool": 0.40, "build": 0.70, "net": 0.0},
        "task_class=debug": {"llm": 0.20, "tool": 0.25, "build": 0.40, "net": 0.0},
        "plan_steps=>8": {"llm": 0.25, "tool": 0.30, "build": 0.10, "net": 0.0},
        "touched_files=>10": {"llm": 0.10, "tool": 0.35, "build": 0.20, "net": 0.0},
        "needs_network=true": {"llm": 0.0, "tool": 0.0, "build": 0.0, "net": 1.6},
    },
}
_INDEX = {c: i for i, c in enumerate(FEATURE_COLUMNS)}


def disagrees_with_prior(truth=HIDDEN_TRUTH) -> dict[str, bool]:
    out = {}
    for op in OPS:
        mu, _ = truth["step"][op]
        ratio = math.exp(mu) / PRIOR_MEDIAN_SECONDS[op]
        lam = math.exp(truth["intercept"][op]) / PRIOR_LAMBDA0[op]
        out[op] = ratio < 0.7 or ratio > 1.4 or lam < 0.7 or lam > 1.4
    return out


def hidden_lambda(op: str, g: dict, truth=HIDDEN_TRUTH) -> float:
    eta = truth["intercept"][op]
    for col, per_op in truth["coef"].items():
        if g.get(col.split("=", 1)[0]) == col.split("=", 1)[1]:
            eta += per_op[op]
        elif col == "needs_network=true" and bool(g.get("needs_network")):
            eta += per_op[op]
    return math.exp(eta)


def _random_g(rng) -> dict:
    return {
        "task_class": str(rng.choice(FEATURE_LEVELS["task_class"])),
        "touched_files": str(rng.choice(FEATURE_LEVELS["touched_files"])),
        "verification_loops": str(rng.choice(FEATURE_LEVELS["verification_loops"])),
        "needs_network": bool(rng.random() < 0.35),
        "plan_steps": str(rng.choice(FEATURE_LEVELS["plan_steps"])),
    }


def _misclassified(rng, g: dict) -> tuple[dict, list]:
    """模拟 LLM 分类：75% 命中真值，错时给一个相邻档位与 top-2 概率。"""
    if rng.random() < CLASSIFY_ACCURACY:
        alt = str(rng.choice([x for x in FEATURE_LEVELS["task_class"] if x != g["task_class"]]))
        return g, [[g["task_class"], 0.65], [alt, 0.35]]
    wrong = str(rng.choice([x for x in FEATURE_LEVELS["task_class"] if x != g["task_class"]]))
    return {**g, "task_class": wrong}, [[wrong, 0.65], [g["task_class"], 0.35]]


def generate_tasks(n_tasks: int, seed: int, truth=HIDDEN_TRUTH):
    rng = np.random.default_rng(seed)
    steps: list[dict] = []
    tasks: list[dict] = []
    clock = 1_730_000_000.0
    for i in range(n_tasks):
        g_true = _random_g(rng)
        g_pred, probs = _misclassified(rng, g_true)
        t0 = clock
        cursor = t0
        rows: list[dict] = []
        for op in OPS:
            for _ in range(rng.poisson(hidden_lambda(op, g_true, truth))):
                mu, sigma = truth["step"][op]
                dur = float(np.exp(rng.normal(mu, sigma)))
                rows.append({"v": 1, "task_id": f"t{i:04d}", "op": op,
                             "t_start": round(cursor, 6), "t_end": round(cursor + dur, 6),
                             "ok": bool(rng.random() < 0.9),
                             "tool": {"llm": "turn", "net": "web", "build": "pytest"}.get(op, "grep")})
                cursor += dur
        full_end = cursor
        censored = bool(rng.random() < CENSORED_RATE)
        if censored and rows:
            cut = max(1, int(len(rows) * float(rng.uniform(0.2, 0.7))))
            rows = rows[:cut]
            cursor = rows[-1]["t_end"]
        steps.extend(rows)
        tasks.append({
            "v": 1, "task_id": f"t{i:04d}", "g_pred": g_pred, "g_pred_probs": probs,
            "g_final": g_true,
            "total_wall": sum(r['t_end'] - r['t_start'] for r in rows),
            "censored": censored,
            "full_end": round(full_end, 6), "model_id": str(rng.choice(MODEL_IDS)),
            "host": HOSTS[0], "estimator_wall": 0.42, "g_schema_version": 1,
        })
        clock = max(clock, cursor) + float(rng.uniform(1.0, 30.0))
    return steps, tasks


def truth_table(env: str = "default"):
    """把 HIDDEN_TRUTH 包成 ConstTable —— 检验'推断能否回收真值'时的标准答案。"""
    import math

    from agent_elapsed.constants import ConstTable, StepParams
    from agent_elapsed.features import FEATURE_COLUMNS, G_SCHEMA_VERSION

    coef = {op: tuple(
        HIDDEN_TRUTH["coef"].get(c, {}).get(op, 0.0) for c in FEATURE_COLUMNS) for op in OPS}
    return ConstTable(
        env=env, g_schema_version=G_SCHEMA_VERSION, fit_id="hidden-truth",
        trained_through_ts=0.0, phi=1.0, task_count=10_000, ks_passed=True,
        intercept={o: HIDDEN_TRUTH["intercept"][o] for o in OPS}, coef=coef,
        step={o: StepParams(mu=HIDDEN_TRUTH["step"][o][0],
                            sigma=HIDDEN_TRUTH["step"][o][1], step_obs=500) for o in OPS},
    )
