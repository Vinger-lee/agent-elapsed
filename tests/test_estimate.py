import unittest

from agent_elapsed.constants import MC_PATHS, OPS, prior_table
from agent_elapsed.estimate import EstResult, estimate_task
from agent_elapsed.features import FeatureError

G = {"task_class": "patch", "touched_files": "2-3", "verification_loops": "2",
     "needs_network": False, "plan_steps": "4-8"}


class EstimateTaskTest(unittest.TestCase):
    def test_contract_has_no_samples_and_no_point_only_number(self):
        r = estimate_task(G, [], table=prior_table(), m=2000, seed=2)
        self.assertEqual(
            set(r.to_dict()),
            {"p50", "p90", "expected", "contributions", "lambdas", "calibrating",
             "fit_id", "seed", "n_paths", "g_schema_version", "p50_se", "p90_se"})
        self.assertNotIn("samples", r.to_dict())
        self.assertLess(r.p50, r.p90)
        # 分位数带着蒙特卡洛不确定度出门，宿主才能判断 5000 条路径够不够
        self.assertGreater(r.p50_se, 0.0)
        self.assertGreater(r.p90_se, 0.0)

    def test_cold_start_reports_calibrating(self):
        self.assertTrue(estimate_task(G, [], table=prior_table(), m=1000).calibrating)

    def test_schema_version_is_carried(self):
        r = estimate_task(G, [], table=prior_table(), m=1000)
        self.assertEqual(r.g_schema_version, 1)
        self.assertEqual(r.n_paths, 1000)

    def test_default_paths_is_5000(self):
        self.assertEqual(estimate_task(G, [], table=prior_table()).n_paths, MC_PATHS)

    def test_deterministic_given_seed(self):
        a = estimate_task(G, [("patch", 1.0)], table=prior_table(), m=1500, seed=99)
        b = estimate_task(G, [("patch", 1.0)], table=prior_table(), m=1500, seed=99)
        self.assertEqual(a.p50, b.p50)
        self.assertEqual(a.seed, 99)

    def test_bad_feature_raises_to_caller(self):
        with self.assertRaises(FeatureError):
            estimate_task({**G, "task_class": "magic"}, [], table=prior_table())

    def test_contributions_cover_every_op(self):
        r = estimate_task(G, [], table=prior_table(), m=1000)
        self.assertEqual(set(r.contributions), set(OPS))
        self.assertEqual(set(r.lambdas), set(OPS))

    def test_is_a_dataclass_of_floats_and_dicts(self):
        r = estimate_task(G, [], table=prior_table(), m=1000)
        self.assertIsInstance(r, EstResult)
        self.assertIsInstance(r.p90, float)
        self.assertIsInstance(r.calibrating, bool)


if __name__ == "__main__":
    unittest.main()
