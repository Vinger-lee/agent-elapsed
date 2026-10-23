import math
import unittest

import numpy as np

from agent_elapsed.constants import OPS, PRIOR_EQUIV_N, PRIOR_SIGMA, StepParams
from agent_elapsed.schema import StepEvent
from offline.fit_step import (
    fit_step_distributions, posterior, prior_for, sigma_hat,
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

    def test_equiv_n_is_the_documented_prior_weight(self):
        self.assertEqual(prior_for("llm").kappa, PRIOR_EQUIV_N)
        self.assertAlmostEqual(math.exp(prior_for("llm").mu), 4.0, places=6)


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
