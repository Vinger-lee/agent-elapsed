"""Regressions for the数学升级: mean-preserving φ, exact dispersion solve,
rank-based quantile intervals, stable NIG scatter and IRLS convergence.

这些测试刻意不依赖上一版实现，全部从定义重算。
"""
from dataclasses import replace
from decimal import Decimal, getcontext
import math
from types import SimpleNamespace
import unittest

import numpy as np

from agent_elapsed.constants import OPS, PHI_MAX, PHI_MIN, StepParams, prior_table
from agent_elapsed.features import FEATURE_COLUMNS
from agent_elapsed.model import (
    cv_squared, expected_total, predict, quantile_interval, quantile_se,
    sample_task, sample_totals, step_moments, task_moments, variance_total,
)
from offline.calibrate import interval_loss, pinball
from offline.fit_counts import fit_ridge_poisson
from offline.fit_step import NIG, posterior, prior_for, sigma_hat
from offline.gate import SEED_BASE, estimate_phi
from offline.synth import generate_tasks, truth_table
from agent_elapsed.schema import parse_task_row

G = {"task_class": "patch", "touched_files": "2-3", "verification_loops": "2",
     "needs_network": False, "plan_steps": "4-8"}


def records(rows):
    out = []
    for r in rows:
        rec, reason = parse_task_row(r)
        assert rec is not None, reason
        out.append(rec)
    return out


class MeanPreservingDispersionTest(unittest.TestCase):
    def test_expected_total_is_independent_of_phi(self):
        base = prior_table()
        rates = {op: 2.0 for op in OPS}
        reference = None
        for phi in (1.0, 2.0, 7.5, 25.0):
            table = replace(base, phi=phi)
            moments = task_moments(table, G, [])
            if reference is None:
                reference = moments["expected"]
            self.assertAlmostEqual(moments["expected"], reference, places=9,
                                   msg=f"phi={phi} 偷改了均值")

    def test_second_moment_is_exactly_mean_squared_times_exp_log_variance(self):
        table = replace(prior_table(), phi=3.0)
        for op in OPS:
            mean, second = step_moments(table, op)
            sigma_eff = table.sigma_used(op)
            self.assertAlmostEqual(second / mean ** 2, math.exp(sigma_eff ** 2), places=12)
            # 与 lognormal 原始定义 exp(2μ_eff + 2σ_eff²) 逐位一致
            mu_eff, _ = table.log_params(op)
            self.assertAlmostEqual(second, math.exp(2 * mu_eff + 2 * sigma_eff ** 2), places=9)

    def test_sampler_second_moment_matches_the_closed_form_under_phi(self):
        # φ=2 keeps the tail light enough that a 120k-path variance estimate
        # converges to ~4%; φ=4 needs ~10x the paths for the same precision.
        table = replace(prior_table(), phi=2.0,
                        step={op: StepParams(math.log(2), math.sqrt(math.log(2)), 200)
                              for op in OPS})
        rates = {op: 3.0 if op == "llm" else 0.0 for op in OPS}
        samples = sample_totals(table, rates, 120_000, 91)
        self.assertAlmostEqual(float(samples.mean()),
                               expected_total(table, rates), delta=.05)
        variance = variance_total(table, rates)
        self.assertAlmostEqual(float(samples.var()), variance,
                               delta=.08 * variance)

    def test_cv_squared_grows_with_phi(self):
        table = prior_table()
        values = [cv_squared(replace(table, phi=phi), G, []) for phi in (1, 2, 4, 8)]
        self.assertEqual(values, sorted(values))
        self.assertGreater(values[-1], values[0])


class DispersionSolveTest(unittest.TestCase):
    def _tasks(self, table, inflate, n, seed):
        """Held-out tasks whose residual variance is inflated by a known factor."""
        samples = sample_task(table, G, [], n, seed)
        return [SimpleNamespace(g_pred=G, g_pred_probs=[], censored=False,
                                g_schema_version=1, total_wall=float(y) * inflate)
                for y in samples]

    def test_solved_phi_matches_the_observed_relative_residual_exactly(self):
        """内点解时，φ 让解析 CV² 与留出残差的均方相对误差完全相等。"""
        table = truth_table()
        tasks = records(generate_tasks(160, seed=61)[1])
        tasks = [t for t in tasks if not t.censored]
        noisy = [SimpleNamespace(g_pred=t.g_pred, g_pred_probs=t.g_pred_probs,
                                 censored=False, g_schema_version=1,
                                 total_wall=t.total_wall * 1.35)
                 for t in tasks]
        phi = estimate_phi(table, noisy)
        self.assertGreater(phi, PHI_MIN)          # 真的解到了内点
        self.assertLess(phi, PHI_MAX)
        solved = replace(table, phi=phi)
        model_cv, residual = [], []
        for task in tasks:
            base = task_moments(table, task.g_pred, list(task.g_pred_probs))
            wide = task_moments(solved, task.g_pred, list(task.g_pred_probs))
            model_cv.append(wide["variance"] / wide["expected"] ** 2)
            residual.append(((task.total_wall * 1.35 - base["expected"])
                             / base["expected"]) ** 2)
        # 解析解与残差的一致性是这套 φ 的全部意义；1e-9 是浮点余量。
        self.assertAlmostEqual(float(np.mean(model_cv)), float(np.mean(residual)),
                               delta=1e-9)

    def test_residual_noise_pushes_phi_up_and_noise_free_stays_at_one(self):
        table = truth_table()
        calm = self._tasks(table, 1.0, 40, 3)
        noisy = self._tasks(table, 3.0, 40, 3)
        self.assertEqual(estimate_phi(table, calm), PHI_MIN)
        self.assertGreater(estimate_phi(table, noisy), PHI_MIN)

    def test_censored_and_foreign_schema_tasks_are_ignored(self):
        table = truth_table()
        tasks = records(generate_tasks(80, seed=63)[1])
        good = [t for t in tasks if not t.censored]
        reference = estimate_phi(table, good)
        padded = good + [SimpleNamespace(g_pred=G, g_pred_probs=[], censored=True,
                                         g_schema_version=1, total_wall=1e9),
                         SimpleNamespace(g_pred=G, g_pred_probs=[], censored=False,
                                         g_schema_version=99, total_wall=1e9)]
        self.assertAlmostEqual(estimate_phi(table, padded), reference, places=12)

    def test_empty_input_returns_the_lower_bound(self):
        self.assertEqual(estimate_phi(truth_table(), []), PHI_MIN)


class QuantileIntervalTest(unittest.TestCase):
    def test_interval_covers_the_true_quantile_at_about_95_percent(self):
        table = truth_table()
        hits = 0
        trials = 200
        true_median = float(np.median(sample_task(table, G, [], 200_000, 19999)))
        for seed in range(trials):
            samples = sample_task(table, G, [], 2000, 1000 + seed)
            lo, hi = quantile_interval(samples, 0.5)
            hits += lo <= true_median <= hi
        self.assertGreater(hits / trials, 0.90)
        self.assertLess(hits / trials, 0.995)

    def test_atom_at_zero_gives_a_degenerate_interval_not_an_error(self):
        samples = np.zeros(500)
        self.assertEqual(quantile_interval(samples, 0.5), (0.0, 0.0))
        self.assertEqual(quantile_se(samples, 0.5), 0.0)

    def test_se_shrinks_when_paths_grow(self):
        table = truth_table()
        small = sample_task(table, G, [], 1000, 7)
        large = sample_task(table, G, [], 16_000, 7)
        self.assertGreater(quantile_se(small, 0.9), quantile_se(large, 0.9))

    def test_predict_reports_the_interval_and_the_se(self):
        out = predict(truth_table(), G, [], m=3000, seed=5)
        lo, hi = out["p90_ci"]
        self.assertLessEqual(lo, out["p90"])
        self.assertLessEqual(out["p90"], hi)
        self.assertAlmostEqual(out["p90_se"], (hi - lo) / (2 * 1.959963984540054), places=12)

    def test_rejects_degenerate_levels(self):
        for tau in (0.0, 1.0, -0.2, 1.2):
            with self.assertRaises(ValueError):
                quantile_interval(np.arange(10.0), tau)


class NIGNumericalStabilityTest(unittest.TestCase):
    @staticmethod
    def _reference(xs, weights, prior: NIG):
        """Exact weighted posterior via Decimal, independent of the numpy path."""
        getcontext().prec = 60
        xx = [Decimal(float(v)) for v in xs]
        ww = [Decimal(float(v)) for v in weights]
        n = sum(ww)
        mu0, kap0 = Decimal(prior.mu), Decimal(prior.kappa)
        beta0 = Decimal(prior.beta)
        mean = sum(w * x for w, x in zip(ww, xx)) / n
        scatter = sum(w * (x - mean) ** 2 for w, x in zip(ww, xx))
        kappa_n = kap0 + n
        dev = mean - mu0
        beta_n = beta0 + scatter / 2 + kap0 * n * dev ** 2 / (2 * kappa_n)
        alpha_n = Decimal(prior.alpha) + n / 2
        return math.sqrt(float(beta_n / (alpha_n - 1)))

    def test_sigma_matches_a_high_precision_reference(self):
        prior = prior_for("llm")
        rng = np.random.default_rng(5)
        xs = prior.mu + rng.normal(0, .4, size=300)
        w = rng.uniform(.2, 1.5, size=300)
        got = sigma_hat(posterior(xs, prior, w))
        self.assertAlmostEqual(got, self._reference(xs, w, prior), places=11)

    def test_scatter_never_uses_the_cancelling_one_pass_shortcut(self):
        """Σx²−(Σx)²/n 的写法在偏移大、离散小时会掉精度；实现不得回落成它。"""
        prior = NIG(mu=1e6, kappa=20.0, alpha=11.0, beta=8.0)
        xs = 1e6 + np.linspace(-3e-4, 3e-4, 200)
        w = np.ones(xs.size)
        reference = self._reference(xs, w, prior)
        stable_sigma = sigma_hat(posterior(xs, prior, w))
        self.assertAlmostEqual(stable_sigma, reference, places=11)
        n = float(w.sum())
        s1 = float(np.dot(w, xs))
        s2 = float(np.dot(w, xs * xs))
        shortcut = max(s2 - s1 * s1 / n, 0.0)
        mean = s1 / n
        kap_n = prior.kappa + n
        beta_n = (prior.beta + 0.5 * shortcut
                  + prior.kappa * n * (mean - prior.mu) ** 2 / (2 * kap_n))
        shortcut_sigma = math.sqrt(beta_n / (prior.alpha + n / 2 - 1.0))
        self.assertLess(abs(stable_sigma - reference), abs(shortcut_sigma - reference) + 1e-12)

    def test_zero_weight_returns_the_prior(self):
        prior = prior_for("tool")
        self.assertEqual(posterior(np.array([1.0, 2.0]), prior, np.zeros(2)), prior)

    def test_non_finite_observations_are_rejected(self):
        prior = prior_for("tool")
        with self.assertRaises(ValueError):
            posterior(np.array([1.0, math.nan]), prior)


class RidgePoissonConvergenceTest(unittest.TestCase):
    def test_gradient_is_zero_at_the_returned_optimum(self):
        rng = np.random.default_rng(11)
        X = np.eye(4)[rng.integers(0, 4, size=500)]
        y = rng.poisson(np.exp(1.0 + X @ np.array([-.5, .2, .8, 0.0])), size=500).astype(float)
        b, w = fit_ridge_poisson(X, y, ridge=1.0)
        p = np.concatenate([[b], w])
        A = np.hstack([np.ones((X.shape[0], 1)), X])
        eta = np.clip(A @ p, -20, 20)
        grad = A.T @ (np.exp(eta) - y)
        grad[1:] += p[1:]
        self.assertLess(float(np.max(np.abs(grad))), 1e-6)

    def test_weighted_start_recovers_the_weighted_rate(self):
        X = np.zeros((4, 1))
        y = np.array([0.0, 0.0, 0.0, 20.0])
        w = np.array([0.0, 0.0, 0.0, 1.0])
        b, _ = fit_ridge_poisson(X, y, ridge=0.0, weights=w)
        self.assertAlmostEqual(math.exp(b), 20.0, delta=1e-6)

    def test_negative_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_ridge_poisson(np.zeros((3, 1)), np.ones(3), weights=np.array([1.0, -1.0, 0.0]))

    def test_intercept_keeps_the_score_equation_under_weights(self):
        """截距不受惩罚 ⟹ Σλ̂ = Σwy / Σw 依然成立（加权版）。"""
        _, rows = generate_tasks(120, seed=71)
        recs = records(rows)
        from agent_elapsed.features import N_FEATURES
        rng = np.random.default_rng(3)
        X = rng.integers(0, 2, size=(len(recs), N_FEATURES)).astype(float)
        y = rng.poisson(5.0, size=len(recs)).astype(float)
        w = rng.uniform(.5, 2.0, size=len(recs))
        b, coef = fit_ridge_poisson(X, y, ridge=1.0, weights=w)
        A = np.hstack([np.ones((len(recs), 1)), X])
        lam = np.exp(np.clip(A @ np.concatenate([[b], coef]), -20, 20))
        wn = w / w.mean()
        self.assertAlmostEqual(float(np.dot(wn, lam)), float(np.dot(wn, y)), places=6)


class LossContractTest(unittest.TestCase):
    def test_interval_loss_is_the_mean_of_the_two_pinballs(self):
        rng = np.random.default_rng(4)
        samples = rng.lognormal(2, .5, size=4000)
        y = 12.0
        expected = (pinball(y, float(np.quantile(samples, .5)), .5)
                    + pinball(y, float(np.quantile(samples, .9)), .9)) / 2.0
        self.assertAlmostEqual(interval_loss(y, samples), expected, places=12)

    def test_fixture_seed_is_stable(self):
        self.assertEqual(SEED_BASE, 20261022)


if __name__ == "__main__":
    unittest.main()
