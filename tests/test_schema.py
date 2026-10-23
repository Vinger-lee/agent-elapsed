import unittest

from agent_elapsed.schema import (
    G_SCHEMA_VERSION, OPS, parse_step_row, parse_task_row,
)


def step_row(**over):
    row = {"v": G_SCHEMA_VERSION, "task_id": "t1", "op": "tool",
           "t_start": 100.0, "t_end": 101.5, "ok": True, "tool": "grep"}
    row.update(over)
    return row


def task_row(**over):
    row = {"v": G_SCHEMA_VERSION, "task_id": "t1",
           "g_pred": {"task_class": "patch"}, "g_pred_probs": [["patch", 0.65], ["debug", 0.35]],
           "g_final": {"task_class": "debug"}, "total_wall": 412.7, "censored": False,
           "model_id": "m-a", "host": "h-a", "estimator_wall": 0.42,
           "g_schema_version": G_SCHEMA_VERSION}
    row.update(over)
    return row


class ThreeBoundaryChecksTest(unittest.TestCase):
    """spec §5.4：校验只做三件事，且违规必须留痕而非静默丢弃。"""

    def test_valid_step_parses(self):
        ev, reason = parse_step_row(step_row(), set())
        self.assertIsNone(reason)
        self.assertAlmostEqual(ev.duration, 1.5)

    def test_op_enum_is_enforced(self):
        ev, reason = parse_step_row(step_row(op="teleport"), set())
        self.assertIsNone(ev)
        self.assertEqual(reason, "bad_op")
        self.assertNotIn("teleport", OPS)

    def test_non_positive_duration_is_rejected(self):
        for bad in (100.0, 99.0, 0.0):
            ev, reason = parse_step_row(step_row(t_end=bad), set())
            self.assertIsNone(ev, f"t_end={bad} 必须被拒")
            self.assertEqual(reason, "non_monotonic")

    def test_duplicate_task_op_start_is_deduped(self):
        seen = set()
        first, _ = parse_step_row(step_row(), seen)
        dup, reason = parse_step_row(step_row(), seen)
        self.assertIsNotNone(first)
        self.assertIsNone(dup)
        self.assertEqual(reason, "duplicate")

    def test_task_parses_and_env_key(self):
        rec, reason = parse_task_row(task_row())
        self.assertIsNone(reason)
        self.assertEqual(rec.env_key(), "m-a|h-a")
        self.assertFalse(rec.censored)


if __name__ == "__main__":
    unittest.main()
