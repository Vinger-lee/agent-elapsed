import json
import tempfile
import unittest
from pathlib import Path

from agent_elapsed.store import TelemetryStore
from tests.test_schema import step_row, task_row


class TelemetryStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.store = TelemetryStore(root / "events.jsonl", root / "tasks.jsonl",
                                    root / "rejects.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_rows_are_appended_and_read_back(self):
        self.assertTrue(self.store.append_step(step_row()))
        self.assertTrue(self.store.append_step(step_row(op="llm", t_start=102.0, t_end=104.0)))
        self.assertTrue(self.store.append_task(task_row()))
        steps = self.store.read_steps()
        self.assertEqual(len(steps), 2)
        self.assertEqual({s.op for s in steps}, {"tool", "llm"})
        self.assertEqual(len(self.store.read_tasks()), 1)

    def test_rejected_rows_are_kept_with_reason_never_dropped(self):
        self.assertFalse(self.store.append_step(step_row(op="bogus")))
        self.assertFalse(self.store.append_step(step_row(t_end=99.0)))
        lines = [json.loads(x) for x in
                 self.store.rejects_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([l["reason"] for l in lines], ["bad_op", "non_monotonic"])
        self.assertEqual(lines[0]["row"]["op"], "bogus")
        self.assertEqual(len(self.store.read_steps()), 0)

    def test_append_is_the_only_write_mode(self):
        self.store.append_step(step_row())
        before = len(self.store.read_steps())
        self.store.append_step(step_row(task_id="t2", t_start=500.0, t_end=501.0))
        self.assertEqual(len(self.store.read_steps()), before + 1)


if __name__ == "__main__":
    unittest.main()
