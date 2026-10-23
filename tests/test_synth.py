import unittest

from agent_elapsed.constants import OPS, PRIOR_LAMBDA0, PRIOR_MEDIAN_SECONDS, prior_table
from agent_elapsed.schema import parse_step_row, parse_task_row
from offline.synth import (
    CENSORED_RATE, HIDDEN_TRUTH, disagrees_with_prior, generate_tasks, hidden_lambda,
)


class NotCircularTest(unittest.TestCase):
    """spec §5.3：生成器必须用与先验表不同的隐藏真值，否则测试必然通过、毫无信息量。"""

    def test_every_op_truth_differs_from_prior_by_a_lot(self):
        flags = disagrees_with_prior()
        self.assertTrue(all(flags.values()), f"这些操作的隐藏真值太接近先验：{flags}")

    def test_sigma_and_lambda0_also_differ(self):
        for op in OPS:
            self.assertNotAlmostEqual(HIDDEN_TRUTH["step"][op][1], 0.8, delta=0.05)
            self.assertGreater(hidden_lambda(op, G_SAMPLE) / PRIOR_LAMBDA0[op], 1.05)


class RowValidityTest(unittest.TestCase):
    def test_generated_rows_all_pass_the_three_boundary_checks(self):
        steps, tasks = generate_tasks(60, seed=1)
        seen = set()
        for row in steps:
            ev, reason = parse_step_row(row, seen)
            self.assertIsNone(reason, f"{reason} @ {row}")
            self.assertGreater(ev.duration, 0)
        for row in tasks:
            rec, reason = parse_task_row(row)
            self.assertIsNone(reason, f"{reason} @ {row}")

    def test_task_count_and_op_set(self):
        steps, tasks = generate_tasks(40, seed=2)
        self.assertEqual(len(tasks), 40)
        self.assertEqual({r["op"] for r in steps}, set(OPS))

    def test_total_wall_equals_step_sum_when_not_censored(self):
        steps, tasks = generate_tasks(50, seed=3)
        by_task = {}
        for row in steps:
            by_task.setdefault(row["task_id"], []).append(row["t_end"] - row["t_start"])
        full = [t for t in tasks if not t["censored"]]
        self.assertTrue(full)
        for t in full:
            self.assertAlmostEqual(t["total_wall"], sum(by_task[t["task_id"]]), places=6)

    def test_censored_tasks_are_truncated_and_tagged(self):
        steps, tasks = generate_tasks(200, seed=4)
        censored = [t for t in tasks if t["censored"]]
        rate = len(censored) / len(tasks)
        self.assertAlmostEqual(rate, CENSORED_RATE, delta=0.08)
        by_task = {}
        for row in steps:
            by_task.setdefault(row["task_id"], []).append(row)
        for t in censored:
            rows = by_task[t["task_id"]]
            self.assertAlmostEqual(t["total_wall"],
                                   sum(r["t_end"] - r["t_start"] for r in rows), places=6)
            self.assertTrue(rows[-1]["t_end"] < t["full_end"])

    def test_model_id_drift_is_representable(self):
        steps, tasks = generate_tasks(30, seed=5)
        self.assertTrue({t["model_id"] for t in tasks})
        self.assertTrue(all(t["host"] for t in tasks))


G_SAMPLE = {"task_class": "patch", "touched_files": "2-3", "verification_loops": "2",
            "needs_network": False, "plan_steps": "4-8"}


if __name__ == "__main__":
    unittest.main()
