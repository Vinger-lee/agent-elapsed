import math
import unittest

import numpy as np

from agent_elapsed.constants import OPS, PRIOR_EQUIV_N, PRIOR_SIGMA, StepParams
from agent_elapsed.schema import StepEvent
from offline.fit_step import (
    LOCATION_PRIOR_N, fit_step_distributions, posterior, prior_for, sigma_hat, summarise,
)
from offline.synth import HIDDEN_TRUTH, generate_tasks


def events(op, durations, task="t"):
    out, cursor = [], 1000.0
    for d in durations:
        out.append(StepEvent(task_id=task, op=op, t_start=cursor, t_end=cursor + d))
        cursor += d
    return out


class ShrinkageTest(unittest.TestCase):
    def test_no_observations_returns_prior_exactly(self):
        params = fit_step_distributions([])
        self.assertEqual(params["llm"].step_obs, 0)
        self.assertAlmostEqual(math.exp(params["llm"].mu), 4.0, places=6)
        self.assertAlmostEqual(params["llm"].sigma, PRIOR_SIGMA, places=6)

    def test_posterior_mean_lies_between_prior_and_mle(self):
        prior = prior_for("llm")
        xs = np.full(8, prior.mu + 2.0)
        post = posterior(xs, prior)
        self.assertGreater(post.mu, prior.mu)
        self.assertLess(post.mu, xs.mean())

    def test_shrinkage_towards_mle_is_monotone_in_sample_size(self):
        prior = prior_for("tool")
        xs = np.full(400, prior.mu + 1.0)
        gaps = [abs(posterior(xs[:n], prior).mu - xs[:n].mean()) for n in (5, 50, 400)]
        self.assertGreater(gaps[0], gaps[1])
        self.assertGreater(gaps[1], gaps[2])

    def test_sigma_hat_recovers_truth_with_many_samples(self):
        rng = np.random.default_rng(0)
        mu, sigma = HIDDEN_TRUTH["step"]["build"]
        xs = rng.normal(mu, sigma, size=3000)
        post = posterior(xs, prior_for("build"))
        self.assertAlmostEqual(post.mu, mu, delta=0.15)
        self.assertAlmostEqual(sigma_hat(post), sigma, delta=0.15)

    def test_tight_steps_are_learned_instead_of_being_hidden_by_the_scale_prior(self):
        """步长比先验紧 40 倍时，几千个观测就该把 σ̂ 压到真值附近。"""
        rng = np.random.default_rng(9)
        true_sigma = 0.02
        xs = math.log(0.05) + rng.normal(0, true_sigma, size=4000)
        prior = prior_for("llm")
        sigma = summarise(xs, prior).sigma_scatter(prior)
        self.assertLess(sigma, 3.0 * true_sigma,
                        f"σ̂={sigma:.3f} 仍被先验撑着，宿主永远过不了覆盖率判据")

    def test_prior_weights_differ_for_location_and_scale_on_purpose(self):
        """位置只给 1 个伪观测、尺度给 PRIOR_EQUIV_N：冷启动中位数是猜测，不能压过数据。"""
        prior = prior_for("llm")
        self.assertEqual(prior.kappa, LOCATION_PRIOR_N)
        self.assertLessEqual(prior.kappa, LOCATION_PRIOR_N)
        self.assertAlmostEqual(prior.alpha, PRIOR_EQUIV_N / 2.0 + 1.0)
        self.assertAlmostEqual(prior.beta, PRIOR_SIGMA ** 2 * (prior.alpha - 1.0))
        self.assertAlmostEqual(math.exp(prior.mu), 4.0, places=6)

    def test_a_wrong_prior_location_does_not_inflate_the_reported_sigma(self):
        """285 个观测都在 0.05 秒附近、先验中位数却写在 4 秒：σ 必须反映数据的离散度。"""
        rng = np.random.default_rng(4)
        xs = math.log(0.05) + rng.normal(0, 0.02, size=285)
        prior = prior_for("llm")
        summary = summarise(xs, prior)
        self.assertGreater(summary.location_gap, 0.0, "先验位置确实错得很远")
        self.assertLess(summary.sigma_scatter(prior), 0.5 * sigma_hat(summary.full),
                        "scatter 口径不该把位置偏差算成离散度")
        self.assertGreater(sigma_hat(summary.full), summary.sigma_scatter(prior),
                           "完整后验确实更大——这正是不能直接拿它预测的原因")
        self.assertLess(abs(summary.full.mu - math.log(0.05)), 0.02,
                        "位置也不该被先验拽回去")


class EndToEndFitTest(unittest.TestCase):
    def test_fit_on_synthetic_steps_returns_mu_sigma_counts(self):
        steps, tasks = generate_tasks(40, seed=12)
        events = [StepEvent(**r) for r in steps]
        params = fit_step_distributions(events)
        self.assertEqual(set(params), set(OPS))
        for op in OPS:
            self.assertIsInstance(params[op], StepParams)
            self.assertGreater(params[op].step_obs, 10)

    def test_recovers_hidden_medians_better_than_prior(self):
        steps, tasks = generate_tasks(120, seed=13)
        events = [StepEvent(**r) for r in steps]
        params = fit_step_distributions(events)
        for op in OPS:
            true_mu, _ = HIDDEN_TRUTH["step"][op]
            prior_mu = prior_for(op).mu
            self.assertLess(abs(params[op].mu - true_mu), abs(prior_mu - true_mu), op)
        self.assertAlmostEqual(math.exp(params["build"].mu), 210.0, delta=60.0)


if __name__ == "__main__":
    unittest.main()
