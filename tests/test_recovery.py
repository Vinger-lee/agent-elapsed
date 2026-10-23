import math
import unittest

import numpy as np

from agent_elapsed.constants import MC_PATHS, OPS, prior_table
from agent_elapsed.model import lambdas, sample_totals
from agent_elapsed.schema import parse_step_row, parse_task_row
from offline.calibrate import calibration_report
from offline.drift import detect, per_task_pinball
from offline.gate import build_candidate, evaluate, temporal_split
from offline.synth import (
    HIDDEN_TRUTH, disagrees_with_prior, generate_tasks, truth_table,
)

G = {"task_class": "patch", "touched_files": "2-3", "verification_loops": "2",
     "needs_network": False, "plan_steps": "4-8"}


def load(n_tasks=900, seed=77):
    step_rows, task_rows = generate_tasks(n_tasks, seed=seed)
    seen, events = set(), []
    for r in step_rows:
        ev, reason = parse_step_row(r, seen)
        assert ev is not None, reason
        events.append(ev)
    records = []
    for r in task_rows:
        rec, reason = parse_task_row(r)
        assert rec is not None, reason
        records.append(rec)
    return events, records


class GuardTest(unittest.TestCase):
    def test_synthetic_truth_is_not_the_prior(self):
        """非循环自证：若生成器=先验，下面所有测试都毫无信息量。"""
        self.assertTrue(all(disagrees_with_prior().values()))
        for op in OPS:
            self.assertGreater(
                abs(math.exp(HIDDEN_TRUTH["step"][op][0])
                    - math.exp(prior_table().step[op].mu)), 0.4, op)


class RecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events, cls.records = load()
        cls.fit, cls.holdout = temporal_split(cls.records)
        cls.cand = build_candidate(cls.events, cls.fit, env="default")

    def test_step_medians_recovered_within_25pct(self):
        for op in OPS:
            truth_mu = HIDDEN_TRUTH["step"][op][0]
            self.assertLess(abs(self.cand.step[op].mu - truth_mu), 0.25, op)
            self.assertGreater(self.cand.step[op].step_obs, 200, op)

    def test_step_sigma_recovered(self):
        for op in OPS:
            truth_sigma = HIDDEN_TRUTH["step"][op][1]
            self.assertLess(abs(self.cand.sigma_used(op) - truth_sigma), 0.3, op)

    def test_lambda_scale_recovered_per_op(self):
        """不逐个比系数，而是比平均期望步数 —— 后者才是真正影响输出的量。"""
        counts = {}
        for ev in self.events:
            per = counts.setdefault(ev.task_id, {o: 0 for o in OPS})
            per[ev.op] += 1
        rows = [r for r in self.fit if not r.censored]
        for op in OPS:
            pred = float(np.mean([
                lambdas(self.cand, r.g_pred, list(r.g_pred_probs))[op] for r in rows]))
            obs = float(np.mean([float(counts[r.task_id][op]) for r in rows]))
            self.assertLess(abs(pred - obs), max(0.35, 0.18 * obs), f"{op}: {pred} vs {obs}")

    def test_fitted_table_beats_the_prior_on_the_same_holdout(self):
        prior_rep = calibration_report(prior_table(), self.holdout, m=1500, seed_base=101)
        cand_rep = calibration_report(self.cand, self.holdout, m=1500, seed_base=101)
        self.assertLess(cand_rep["pinball"]["mean"], prior_rep["pinball"]["mean"])

    def test_gate_accepts_the_fitted_candidate(self):
        verdict = evaluate(prior_table(), self.cand, self.holdout)
        self.assertTrue(verdict.accepted, msg=str(verdict.details["report"]))

    def test_p90_band_covers_about_90_percent(self):
        rep = calibration_report(truth_table(), self.records, m=2000, seed_base=103)
        cov = rep["taus"]["0.9"]["coverage"]
        self.assertLess(abs(cov - 0.9), rep["taus"]["0.9"]["tol"])

    def test_no_drift_alert_on_a_stationary_stream(self):
        scores = per_task_pinball(truth_table(), self.records, m=800, seed_base=104)
        self.assertFalse(detect(scores, baseline_n=50)["alert"])

    def test_cold_start_output_is_a_usable_band_not_garbage(self):
        t = prior_table()
        s = sample_totals(t, lambdas(t, G, []), m=MC_PATHS, seed=7)
        p50 = float(np.quantile(s, 0.5))
        p90 = float(np.quantile(s, 0.9))
        self.assertGreater(p50, 20.0)
        self.assertLess(p50, 600.0)
        self.assertGreater(p90, p50)

    def test_runtime_estimate_matches_offline_recomputation_exactly(self):
        """同一种子下运行时与离线复算必须逐位一致 —— 否则线上数字无法被离线复现。"""
        from agent_elapsed.estimate import estimate_task

        online = estimate_task(G, [("patch", 1.0)], table=self.cand, m=2000, seed=555)
        s = sample_totals(self.cand, lambdas(self.cand, G, [("patch", 1.0)]), 2000, 555)
        self.assertEqual(online.p50, float(np.quantile(s, 0.5)))
        self.assertEqual(online.p90, float(np.quantile(s, 0.9)))


if __name__ == "__main__":
    unittest.main()
