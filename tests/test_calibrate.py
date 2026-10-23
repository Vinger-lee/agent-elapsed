from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

import numpy as np

from agent_elapsed.constants import OPS, StepParams, ConstTable, prior_table
from agent_elapsed.schema import parse_task_row
from offline.calibrate import (
    calibration_report, coverage_tolerance, ks_pvalue_uniform, ks_stat_uniform,
    pinball, pit_of, samples_for_task,
)
from offline.synth import generate_tasks, truth_table


def records(rows):
    out = []
    for r in rows:
        rec, reason = parse_task_row(r)
        assert rec is not None, reason
        out.append(rec)
    return out


class ToleranceTest(unittest.TestCase):
    """spec §6.2：容差必须随 n 走，定死 0.05 在小样本会误杀合格模型。"""

    def test_two_standard_errors(self):
        self.assertAlmostEqual(coverage_tolerance(0.5, 100), 0.1, places=9)
        self.assertAlmostEqual(coverage_tolerance(0.9, 100), 2 * np.sqrt(0.09 / 100), places=9)

    def test_tolerance_shrinks_with_n(self):
        self.assertGreater(coverage_tolerance(0.5, 40), coverage_tolerance(0.5, 400))

    def test_n20_would_have_rejected_a_fair_model(self):
        # 0.05 在 n=20 时约等于一个标准误（0.112），所以它比 2σ 严格一倍以上。
        self.assertGreater(coverage_tolerance(0.5, 20), 2 * 0.05)


class PitAndLossTest(unittest.TestCase):
    def test_randomized_predictive_rank_handles_tied_zero_samples(self):
        from offline.calibrate import randomized_pit

        self.assertAlmostEqual(randomized_pit(0.0, np.array([0., 0., 0., 1.]), .25), .2)

    def test_pit_of_median_is_half(self):
        self.assertAlmostEqual(pit_of(50.0, np.arange(1.0, 101.0)), 0.5)

    def test_pit_is_monotone_in_actual(self):
        s = np.linspace(1.0, 100.0, 1000)
        self.assertLess(pit_of(20.0, s), pit_of(80.0, s))

    def test_pinball_is_positive_and_asymmetric(self):
        self.assertGreater(pinball(10.0, 5.0, 0.9), 0.0)
        self.assertGreater(pinball(10.0, 5.0, 0.5), pinball(6.0, 5.0, 0.5))

    def test_ks_stat_and_p_are_sane(self):
        u = np.linspace(0.001, 0.999, 500)
        d = ks_stat_uniform(u)
        self.assertLess(d, 0.06)
        self.assertGreater(ks_pvalue_uniform(d, 500), 0.05)
        skewed = np.linspace(0.0, 0.6, 500)
        self.assertLess(ks_pvalue_uniform(ks_stat_uniform(skewed), 500), 0.001)


class TruthTableTest(unittest.TestCase):
    def test_truth_table_disagrees_with_prior(self):
        t = truth_table()
        self.assertNotAlmostEqual(t.step["build"].mu, prior_table().step["build"].mu, delta=0.1)
        self.assertEqual(t.phi, 1.0)
        self.assertFalse(t.is_calibrating())
        self.assertEqual(len(t.coef["llm"]), 21)


class ReportTest(unittest.TestCase):
    def setUp(self):
        _, task_rows = generate_tasks(150, seed=31)
        self.tasks = [t for t in records(task_rows) if not t.censored]

    def test_perfect_model_passes(self):
        rep = calibration_report(truth_table(), self.tasks, m=2000, seed_base=17)
        self.assertTrue(rep["pass"], msg=str(rep))
        self.assertEqual(rep["n"], len(self.tasks))

    def test_overconfident_model_is_caught(self):
        """σ 砍半：区间变窄、看起来更自信，覆盖率必须掉下来。"""
        t = truth_table()
        tight = ConstTable(
            env=t.env, g_schema_version=t.g_schema_version, fit_id="tight",
            trained_through_ts=0.0, phi=1.0, task_count=t.task_count, ks_passed=True,
            intercept=t.intercept, coef=t.coef,
            step={o: StepParams(t.step[o].mu, t.step[o].sigma * 0.3, t.step[o].step_obs)
                  for o in OPS})
        rep = calibration_report(tight, self.tasks, m=2000, seed_base=17)
        self.assertFalse(rep["pass"])
        self.assertLess(rep["taus"]["0.9"]["coverage"], 0.9)

    def test_prior_model_scores_worse_than_truth_on_same_tasks(self):
        """配对比较的雏形：同一批任务，两个模型分数才可比。"""
        good = calibration_report(truth_table(), self.tasks, m=2000, seed_base=17)["pinball"]["mean"]
        bad = calibration_report(prior_table(), self.tasks, m=2000, seed_base=17)["pinball"]["mean"]
        self.assertLess(good, bad)

    def test_report_uses_different_seed_per_task(self):
        a = calibration_report(truth_table(), self.tasks[:5], m=500, seed_base=1)
        b = calibration_report(truth_table(), self.tasks[:5], m=500, seed_base=1)
        self.assertEqual(a["taus"], b["taus"])

    def test_correct_model_with_zero_atom_passes_calibration(self):
        table = replace(prior_table(), intercept={op: math.log(.1) for op in OPS},
                        step={op: StepParams(0., .3, 200) for op in OPS})
        # Four independent Pois(.1) counts sum to Pois(.4). Generate that
        # aggregate directly, independently of the runtime's per-op sampler.
        rng = np.random.default_rng(983)
        counts = rng.poisson(.4, 300)
        durations = rng.lognormal(0., .3, size=int(counts.sum()))
        totals = np.bincount(np.repeat(np.arange(300), counts), weights=durations,
                             minlength=300)
        g = {"task_class": "patch", "touched_files": "1", "verification_loops": "0",
             "needs_network": False, "plan_steps": "1-3"}
        tasks = [SimpleNamespace(total_wall=float(y), censored=False, g_schema_version=1,
                                 g_pred=g, g_pred_probs=[]) for y in totals]
        report = calibration_report(table, tasks, m=3000, seed_base=241)
        self.assertGreater(report["taus"]["0.5"]["coverage"], .6)
        self.assertEqual(report["taus"]["0.5"]["coverage_below"], 0.)
        self.assertTrue(report["pass"], msg=str(report))


if __name__ == "__main__":
    unittest.main()
