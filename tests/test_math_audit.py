"""Independent closed-form checks for the duration distribution, not real-world accuracy."""
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

import numpy as np

from agent_elapsed import model
from agent_elapsed.constants import OPS, StepParams, prior_table
from agent_elapsed.features import FEATURE_COLUMNS
from offline.calibrate import samples_for_task
from offline.gate import estimate_phi

G = {"task_class": "patch", "touched_files": "1", "verification_loops": "0",
     "needs_network": False, "plan_steps": "1-3"}
PROBS = [("patch", .5), ("debug", .5)]


def two_class_table():
    # Every jump lasts exactly 1s. Four operations each have rate 1 or 9,
    # so T|patch ~ Pois(4), T|debug ~ Pois(36).
    table = prior_table()
    coef = {op: list(table.coef[op]) for op in OPS}
    for op in OPS:
        coef[op][FEATURE_COLUMNS.index("task_class=debug")] = math.log(9)
    return replace(table, intercept={op: 0.0 for op in OPS},
                   coef={op: tuple(values) for op, values in coef.items()},
                   step={op: StepParams(0.0, 0.0, 200) for op in OPS})


class ConditionalMomentsTest(unittest.TestCase):
    def test_closed_form_includes_poisson_count_variation(self):
        # mu=ln(2), sigma^2=ln(2): E[X]=2sqrt(2), E[X^2]=16.
        # With Pois(3) jumps, E[T]=6sqrt(2) and Var(T)=48 (not 24).
        table = replace(prior_table(), step={
            op: StepParams(math.log(2), math.sqrt(math.log(2)), 200) for op in OPS})
        rates = {op: 3.0 if op == "llm" else 0.0 for op in OPS}
        self.assertAlmostEqual(model.expected_total(table, rates), 6 * math.sqrt(2))
        self.assertAlmostEqual(model.variance_total(table, rates), 48.0)

    def test_phi_scales_only_the_second_moment(self):
        # mu=ln(2), sigma^2=ln(2): E[X]=2sqrt(2) for every phi.
        # phi=2 only inflates E[X²] from 16 to E[X]²·exp(2sigma²)=32,
        # so E[T] stays 6sqrt(2) and Var(T) doubles from 48 to 96.
        table = replace(prior_table(), phi=2, step={
            op: StepParams(math.log(2), math.sqrt(math.log(2)), 200) for op in OPS})
        rates = {op: 3.0 if op == "llm" else 0.0 for op in OPS}
        self.assertAlmostEqual(model.expected_total(table, rates), 6 * math.sqrt(2))
        self.assertAlmostEqual(model.variance_total(table, rates), 96.0)

    def test_phi_keeps_the_lognormal_mean_in_the_sampler(self):
        """φ 是纯离散度参数：样本均值不随 φ 移动，分散度随 φ 上升。"""
        base = replace(prior_table(), step={
            op: StepParams(math.log(2), math.sqrt(math.log(2)), 200) for op in OPS})
        wide = replace(base, phi=4)
        rates = {op: 3.0 if op == "llm" else 0.0 for op in OPS}
        narrow_samples = model.sample_totals(base, rates, 60_000, 71)
        wide_samples = model.sample_totals(wide, rates, 60_000, 71)
        self.assertAlmostEqual(float(wide_samples.mean()),
                               float(narrow_samples.mean()), delta=.25)
        self.assertGreater(float(wide_samples.var()), 2.0 * float(narrow_samples.var()))

    def test_sampler_matches_independently_derived_mean_and_variance(self):
        table = replace(prior_table(), step={
            op: StepParams(math.log(2), math.sqrt(math.log(2)), 200) for op in OPS})
        rates = {op: 3.0 if op == "llm" else 0.0 for op in OPS}
        samples = model.sample_totals(table, rates, 100_000, 73)
        self.assertAlmostEqual(float(samples.mean()), 6 * math.sqrt(2), delta=.12)
        self.assertAlmostEqual(float(samples.var()), 48.0, delta=3.0)


class TaskClassMixtureTest(unittest.TestCase):
    def test_one_class_draw_per_task_preserves_between_class_variation(self):
        samples = model.predict(two_class_table(), G, PROBS, m=40_000, seed=14)["samples"]
        # The 50/50 mixture has mean 20, within-class variance 20, and
        # between-class variance (36-4)^2/4 = 256. Total variance is 276.
        # A Pois(20) rate average has variance 20 and puts ~1% below 10s;
        # the correct task-level mixture puts nearly half below 10s.
        self.assertAlmostEqual(float(samples.mean()), 20.0, delta=.3)
        self.assertAlmostEqual(float(samples.var()), 276.0, delta=8.0)
        self.assertGreater(float(np.mean(samples <= 10)), .45)
        self.assertLess(float(np.mean(samples <= 10)), .55)

    def test_closed_form_keeps_within_and_between_class_variance(self):
        moments = model.task_moments(two_class_table(), G, PROBS)
        self.assertAlmostEqual(moments["expected"], 20.0)
        self.assertAlmostEqual(moments["within_class_variance"], 20.0)
        self.assertAlmostEqual(moments["between_class_variance"], 256.0)
        self.assertAlmostEqual(moments["variance"], 276.0)

    def test_large_class_weights_preserve_moments_and_sampling(self):
        table = two_class_table()
        large = [("patch", 1e308), ("debug", 1e308)]
        self.assertEqual(model.task_moments(table, G, large),
                         model.task_moments(table, G, PROBS))
        self.assertTrue(np.array_equal(model.sample_task(table, G, large, 4000, 25),
                                       model.sample_task(table, G, PROBS, 4000, 25)))

    def test_runtime_and_calibration_use_same_task_distribution(self):
        table = two_class_table()
        task = SimpleNamespace(g_pred=G, g_pred_probs=PROBS)
        runtime = model.predict(table, G, PROBS, m=4000, seed=25)["samples"]
        offline = samples_for_task(table, task, m=4000, seed=25)
        self.assertTrue(np.array_equal(runtime, offline))

    def test_phi_does_not_mislabel_class_uncertainty_as_extra_noise(self):
        task = SimpleNamespace(g_pred=G, g_pred_probs=PROBS, censored=False,
                               g_schema_version=1, total_wall=20 + math.sqrt(276))
        self.assertAlmostEqual(estimate_phi(two_class_table(), [task]), 1.0)


if __name__ == "__main__":
    unittest.main()
