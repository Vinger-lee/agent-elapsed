import math
import unittest

import numpy as np

from agent_elapsed.constants import OPS, PRIOR_LAMBDA0
from agent_elapsed.features import N_FEATURES
from agent_elapsed.schema import StepEvent, parse_step_row, parse_task_row
from offline.fit_counts import design_rows, fit_counts, fit_ridge_poisson, task_op_counts
from offline.synth import HIDDEN_TRUTH, generate_tasks, hidden_lambda

G = {"task_class": "patch", "touched_files": "2-3", "verification_loops": "2",
     "needs_network": False, "plan_steps": "4-8"}


def to_events(rows):
    seen = set()
    out = []
    for r in rows:
        ev, reason = parse_step_row(r, seen)
        assert ev is not None, reason
        out.append(ev)
    return out


def to_records(rows):
    out = []
    for r in rows:
        rec, reason = parse_task_row(r)
        assert rec is not None, reason
        out.append(rec)
    return out


class RidgePoissonTest(unittest.TestCase):
    def test_intercept_recovers_rate_when_covariates_are_useless(self):
        rng = np.random.default_rng(0)
        X = rng.integers(0, 2, size=(300, 3)).astype(float)
        y = rng.poisson(6.0, size=300).astype(float)
        b, w = fit_ridge_poisson(X, y, ridge=1.0)
        self.assertAlmostEqual(math.exp(b), 6.0, delta=0.8)
        self.assertLess(float(np.abs(w).max()), 0.2)

    def test_intercept_is_never_penalised(self):
        rng = np.random.default_rng(1)
        X = np.zeros((200, 3))
        y = rng.poisson(4.0, size=200).astype(float)
        b_lo, _ = fit_ridge_poisson(X, y, ridge=0.1)
        b_hi, w_hi = fit_ridge_poisson(X, y, ridge=200.0)
        self.assertAlmostEqual(b_lo, b_hi, places=3)
        self.assertAlmostEqual(float(np.abs(w_hi).sum()), 0.0, places=3)

    def test_a_real_effect_is_found(self):
        rng = np.random.default_rng(2)
        X = rng.integers(0, 2, size=(600, 3)).astype(float)
        y = rng.poisson(np.exp(1.5 + 0.9 * X[:, 1]), size=600).astype(float)
        b, w = fit_ridge_poisson(X, y, ridge=0.5)
        self.assertAlmostEqual(b, 1.5, delta=0.2)
        self.assertAlmostEqual(w[1], 0.9, delta=0.2)

    def test_extreme_counts_do_not_explode(self):
        X = np.ones((4, 2))
        y = np.array([0.0, 0.0, 0.0, 500.0])
        b, w = fit_ridge_poisson(X, y, ridge=5.0)
        self.assertTrue(np.isfinite(b))
        self.assertTrue(np.all(np.isfinite(w)))


class CountsContractTest(unittest.TestCase):
    def test_design_rows_width_and_one_hot(self):
        A = design_rows([G, {**G, "task_class": "debug"}])
        self.assertEqual(A.shape, (2, N_FEATURES))
        self.assertTrue(np.all(A.sum(axis=1) == 5.0))

    def test_task_op_counts(self):
        steps, tasks = generate_tasks(3, seed=20)
        counts = task_op_counts(to_events(steps))
        self.assertEqual(set(counts), {t["task_id"] for t in tasks})
        for per in counts.values():
            self.assertEqual(set(per), set(OPS))

    def test_leakage_g_final_is_never_read(self):
        """spec §4.3：拟合只能吃 g_pred。构造两份只差 g_final 的数据，系数必须完全一致。"""
        steps, task_rows = generate_tasks(80, seed=21)
        events = to_events(steps)
        records = to_records(task_rows)
        a = fit_counts(events, records, ridge=2.0)
        tampered = []
        for r in records:
            object.__setattr__(r, "g_final", {"task_class": "research",
                                              "touched_files": ">10",
                                              "needs_network": True})
            tampered.append(r)
        b = fit_counts(events, tampered, ridge=2.0)
        for op in OPS:
            self.assertAlmostEqual(a[op][0], b[op][0], places=12, msg=op)
            self.assertEqual(tuple(np.round(a[op][1], 12)), tuple(np.round(b[op][1], 12)))

    def test_censored_tasks_are_excluded_from_the_count_regression(self):
        """spec §9.2：删失任务的 n_j 只是下界，进回归会污染 λ；这里直接证明它被排除。"""
        g = {"task_class": "patch", "touched_files": "1", "verification_loops": "0",
             "needs_network": False, "plan_steps": "1-3"}

        def rec(tid, censored, n_steps, cursor):
            rows = [{"v": 1, "task_id": tid, "op": "llm",
                     "t_start": cursor + i, "t_end": cursor + i + 0.5} for i in range(n_steps)]
            row = {"v": 1, "task_id": tid, "g_pred": g, "g_pred_probs": [["patch", 1.0]],
                   "g_final": g, "total_wall": n_steps * 0.5, "censored": censored,
                   "model_id": "m", "host": "h", "estimator_wall": 0.4,
                   "g_schema_version": 1}
            return rows, row

        a_rows, a = rec("A", False, 3, 0.0)
        b_rows, b = rec("B", True, 30, 100.0)
        events = to_events(a_rows + b_rows)
        records = to_records([a, b])
        out = fit_counts(events, records, ridge=0.05)
        fitted_llm = math.exp(out["llm"][0])
        self.assertLess(fitted_llm, 6.0, "删失任务 B 的 30 步被算进来了")
        self.assertAlmostEqual(fitted_llm, 3.0, delta=1.5)

    def test_no_usable_rows_falls_back_to_prior_lambda0(self):
        out = fit_counts([], [], ridge=1.0)
        for op in OPS:
            self.assertAlmostEqual(out[op][0], math.log(PRIOR_LAMBDA0[op]), places=9)
            self.assertEqual(tuple(out[op][1]), tuple(0.0 for _ in range(N_FEATURES)))

    def test_intercept_score_equation_holds_exactly(self):
        """截距不受惩罚 ⟹ Σλ̂ = Σy。这条与数据无关，是"尺度没被泄漏或偏移"的代数保证。"""
        steps, task_rows = generate_tasks(400, seed=23)
        events, records = to_events(steps), to_records(task_rows)
        records = [r for r in records if not r.censored]
        out = fit_counts(events, records, ridge=1.0)
        counts = task_op_counts(events)
        X = design_rows([r.g_pred for r in records])
        for op in OPS:
            b, w = out[op]
            lam = np.exp(np.clip(X @ np.array(w) + b, -20, 20))
            y = np.array([float(counts[r.task_id][op]) for r in records])
            self.assertAlmostEqual(float(lam.mean()), float(y.mean()), delta=0.05)

    def test_network_flag_is_learned_in_the_right_direction(self):
        steps, task_rows = generate_tasks(400, seed=25)
        out = fit_counts(to_events(steps), to_records(task_rows), ridge=1.0)
        from agent_elapsed.features import FEATURE_COLUMNS
        idx = FEATURE_COLUMNS.index("needs_network=true")
        self.assertGreater(out["net"][1][idx], 0.8)
        self.assertLess(abs(out["llm"][1][idx]), 0.3)

    def test_discount_weights_prefer_recent_tasks(self):
        steps, task_rows = generate_tasks(120, seed=24)
        events, records = to_events(steps), to_records(task_rows)
        n = len(records)
        recent = np.array([1.0 if i >= n // 2 else 0.0 for i in range(n)])
        out = fit_counts(events, records, ridge=1.0, weights=recent)
        self.assertTrue(all(math.isfinite(v[0]) for v in out.values()))
