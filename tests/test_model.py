import unittest

import numpy as np

from agent_elapsed.constants import OPS, prior_table
from agent_elapsed.model import (
    expected_by_op, expected_total, lambdas, predict, sample_totals,
)

G = {"task_class": "patch", "touched_files": "2-3", "verification_loops": "2",
     "needs_network": False, "plan_steps": "4-8"}


def lam_of(table, g=G, probs=()):
    return lambdas(table, g, list(probs))


class LambdaTest(unittest.TestCase):
    def test_zero_coef_gives_lambda0(self):
        t = prior_table()
        lam = lam_of(t)
        self.assertAlmostEqual(lam["llm"], 8.0, places=6)
        self.assertAlmostEqual(lam["build"], 1.5, places=6)

    def test_coef_on_one_hot_column_moves_lambda(self):
        t = prior_table()
        from agent_elapsed.features import FEATURE_COLUMNS
        idx = FEATURE_COLUMNS.index("task_class=patch")
        coef = {o: tuple(0.0 for _ in range(len(FEATURE_COLUMNS))) for o in OPS}
        c = list(coef["llm"]); c[idx] = 0.5; coef["llm"] = tuple(c)
        moved = ConstWithCoef(t, coef)
        self.assertAlmostEqual(lam_of(moved)["llm"], 8.0 * np.exp(0.5), places=5)

    def test_soft_mix_is_probability_weighted(self):
        t = prior_table()
        from agent_elapsed.features import FEATURE_COLUMNS
        coef = {o: list(t.coef[o]) for o in OPS}
        coef["tool"][FEATURE_COLUMNS.index("task_class=debug")] = 1.0
        coef = {o: tuple(v) for o, v in coef.items()}
        mixed = ConstWithCoef(t, coef)
        one = lam_of(mixed, {**G, "task_class": "debug"}, ())["tool"]
        mix = lam_of(mixed, G, [("patch", 0.65), ("debug", 0.35)])["tool"]
        self.assertAlmostEqual(mix, 12.0 * 0.65 + one * 0.35, places=5)


class MomentsTest(unittest.TestCase):
    def test_expected_matches_lognormal_mean(self):
        t = prior_table()
        lam = lam_of(t)
        self.assertAlmostEqual(
            expected_by_op(t, lam)["llm"], 8.0 * np.exp(t.step["llm"].mu + 0.8 ** 2 / 2),
            places=6)
        self.assertAlmostEqual(expected_total(t, lam),
                               sum(expected_by_op(t, lam).values()), places=6)


class SampleTotalsTest(unittest.TestCase):
    def test_deterministic_for_a_seed(self):
        t = prior_table()
        lam = lam_of(t)
        a = sample_totals(t, lam, m=4000, seed=7)
        b = sample_totals(t, lam, m=4000, seed=7)
        self.assertTrue(np.array_equal(a, b))

    def test_median_is_within_20pct_of_mean(self):
        """右偏和的 P50 必低于 E[T]，但不应偏离得离谱——这条守住单位/倍率错误。"""
        t = prior_table()
        lam = lam_of(t)
        s = sample_totals(t, lam, m=8000, seed=11)
        e = expected_total(t, lam)
        p50 = float(np.quantile(s, 0.5))
        self.assertLess(p50, e)
        self.assertGreater(p50, 0.55 * e)

    def test_zero_lambda_yields_exactly_zero_time(self):
        t = prior_table()
        s = sample_totals(t, {o: 0.0 for o in OPS}, m=500, seed=3)
        self.assertEqual(float(s.sum()), 0.0)

    def test_phi_widens_the_distribution_without_moving_the_mean(self):
        """φ 展宽是保均值的一阶随机占优：均值不动，两端变宽。"""
        t = prior_table()
        lam = lam_of(t)
        narrow = sample_totals(t, lam, m=6000, seed=5)
        wide = sample_totals(ConstWithPhi(t, 4.0), lam, m=6000, seed=5)
        self.assertGreater(float(np.quantile(wide, 0.9)), float(np.quantile(narrow, 0.9)))
        self.assertGreater(float(np.std(wide)), float(np.std(narrow)))
        self.assertAlmostEqual(float(wide.mean()), float(narrow.mean()),
                               delta=.02 * float(narrow.mean()))


class PredictContractTest(unittest.TestCase):
    def test_contributions_sum_to_one_and_quantiles_ordered(self):
        out = predict(prior_table(), G, [("patch", 0.65), ("debug", 0.35)], m=3000, seed=1)
        self.assertAlmostEqual(sum(out["contributions"].values()), 1.0, places=9)
        self.assertLessEqual(out["p50"], out["p90"])
        self.assertEqual(set(out["contributions"]), set(OPS))
        self.assertEqual(out["samples"].shape, (3000,))


def ConstWithPhi(t, phi):
    from agent_elapsed.constants import ConstTable
    return ConstTable(env=t.env, g_schema_version=t.g_schema_version, fit_id=t.fit_id,
                      trained_through_ts=t.trained_through_ts, phi=phi,
                      task_count=t.task_count, ks_passed=t.ks_passed,
                      intercept=t.intercept, coef=t.coef, step=t.step)


def ConstWithCoef(t, coef):
    from agent_elapsed.constants import ConstTable
    return ConstTable(env=t.env, g_schema_version=t.g_schema_version, fit_id=t.fit_id,
                      trained_through_ts=t.trained_through_ts, phi=t.phi,
                      task_count=t.task_count, ks_passed=t.ks_passed,
                      intercept=t.intercept, coef=coef, step=t.step)
