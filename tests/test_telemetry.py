"""Recorder tests: the hook lifecycle must actually become telemetry, once and correctly."""
import tempfile
import unittest
from pathlib import Path

from agent_elapsed.constants import prior_table
from agent_elapsed.hooks import DisplayPolicy, ElapsedHooks
from agent_elapsed.store import TelemetryStore
from agent_elapsed.telemetry import TelemetryRecorder
from offline.partition import partition_tasks

G = {"task_class": "patch", "touched_files": "2-3", "verification_loops": "2",
     "needs_network": False, "plan_steps": "4-8"}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class RecorderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.store = TelemetryStore(root / "events.jsonl", root / "tasks.jsonl",
                                    root / "rejects.jsonl")
        self.clock = Clock()
        self.hooks = ElapsedHooks(prior_table(), policy=DisplayPolicy(long_task_seconds=5),
                                  clock=self.clock)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.hooks.close)
        self.recorder = TelemetryRecorder(self.hooks, self.store, model_id="m-a", host="h-a",
                                          clock=self.clock)

    def test_finished_task_is_recorded_with_features_and_estimate(self):
        self.recorder.start("t1", "修复接口", g=G, probs=[("patch", .7), ("debug", .3)])
        self.clock.advance(30)
        outcome = self.recorder.finish("t1")
        self.assertEqual(outcome.status, "completed")

        tasks = self.store.read_tasks()
        self.assertEqual(len(tasks), 1)
        record = tasks[0]
        self.assertEqual(record.task_id, "t1")
        self.assertEqual(record.g_pred, G)
        self.assertEqual(record.g_pred_probs, (("patch", .7), ("debug", .3)))
        self.assertFalse(record.censored)
        self.assertAlmostEqual(record.total_wall, 30.0)
        self.assertEqual(record.env_key(), "m-a|h-a")

    def test_cancelled_task_is_censored_and_failed_task_is_not(self):
        self.recorder.start("c", "取消我", g=G)
        self.clock.advance(10)
        self.recorder.cancel("c")
        self.recorder.start("f", "让我失败", g=G)
        self.clock.advance(10)
        self.recorder.fail("f")

        by_id = {t.task_id: t for t in self.store.read_tasks()}
        self.assertTrue(by_id["c"].censored, "取消任务的总时长只是下界，必须记成删失")
        self.assertFalse(by_id["f"].censored, "失败任务确实结束了，是完整观测")

    def test_steps_are_written_immediately_with_their_durations(self):
        self.recorder.start("t2", "多步任务", g=G)
        with self.recorder.step("t2", "llm"):
            self.clock.advance(2)
        with self.recorder.step("t2", "tool", tool="grep") as state:
            self.clock.advance(3)
            state["ok"] = False
        self.clock.advance(1)
        self.recorder.finish("t2")

        steps = self.store.read_steps()
        self.assertEqual([s.op for s in steps], ["llm", "tool"])
        self.assertAlmostEqual(steps[0].duration, 2.0)
        self.assertAlmostEqual(steps[1].duration, 3.0)
        self.assertTrue(steps[0].ok)
        self.assertFalse(steps[1].ok, "失败步骤保留时长，但 ok 必须是 False")
        self.assertEqual(steps[1].tool, "grep")

    def test_steps_survive_a_task_that_never_finishes(self):
        """宿主崩了/进程被杀时，已发生的步骤本身就是完整观测。"""
        self.recorder.start("t3", "中途崩", g=G)
        with self.recorder.step("t3", "build"):
            self.clock.advance(4)
        self.assertEqual(len(self.store.read_steps()), 1)
        self.assertEqual(self.store.read_tasks(), [])

    def test_repeated_termination_writes_one_task_record(self):
        self.recorder.start("t4", "重复收尾", g=G)
        self.clock.advance(5)
        self.assertIsNotNone(self.recorder.finish("t4"))
        self.assertIsNone(self.recorder.finish("t4"))
        self.assertIsNone(self.recorder.cancel("t4"))
        self.assertEqual(len(self.store.read_tasks()), 1)

    def test_task_without_features_is_still_recorded_and_then_quarantined(self):
        """没有分类也要落盘：墙钟时间是真数据，缺列由离线侧隔离而不是丢掉记录。"""
        self.recorder.start("t5", "没分类")
        self.clock.advance(7)
        self.recorder.finish("t5")

        tasks = self.store.read_tasks()
        self.assertEqual(len(tasks), 1, "记录必须写进去")
        partition = partition_tasks(tasks, prior_table(), complete_only=True)
        self.assertEqual(len(partition.usable), 0, "缺列记录不能进拟合")
        self.assertIn("features:", partition.skipped[0][1])

    def test_bad_model_identity_is_refused_early(self):
        for bad in ("", "   ", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                TelemetryRecorder(self.hooks, self.store, model_id=bad or "", host="h-a")

    def test_store_rejection_never_raises_at_the_host(self):
        """时间倒流之类的坏行只返回 False，不允许把异常抛回宿主。"""
        self.assertFalse(self.store.append_step({"v": 1, "task_id": "x", "op": "llm",
                                                 "t_start": 5.0, "t_end": 1.0}))
        self.assertEqual(len(self.store.read_steps()), 0)


if __name__ == "__main__":
    unittest.main()
