"""Lifecycle tests: display only task work, and never revive a finished task."""
import threading
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from agent_elapsed.constants import prior_table
from agent_elapsed.estimate import EstResult
from agent_elapsed.presentation import format_expectation
from agent_elapsed.hooks import DisplayPolicy, ElapsedHooks


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def prediction(*, calibrating=False, p50=90, p90=180):
    return EstResult(p50=p50, p90=p90, expected=120,
                     contributions={"llm": .5, "tool": .5}, lambdas={},
                     calibrating=calibrating, fit_id="fixture", seed=7,
                     n_paths=5000, g_schema_version=1)


G = {"task_class": "patch", "touched_files": "2-3",
     "verification_loops": "2", "needs_network": False, "plan_steps": "4-8"}


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.hooks = ElapsedHooks(prior_table(), clock=self.clock)
        self.addCleanup(self.hooks.close)

    def test_unknown_length_task_stays_hidden_until_sixty_seconds(self):
        self.hooks.on_task_start("a", "修复登录")
        self.clock.advance(59)
        self.assertEqual(self.hooks.refresh().to_dict()["tasks"], [])
        self.clock.advance(1)
        task = self.hooks.refresh().to_dict()["tasks"][0]
        self.assertEqual(task["task_id"], "a")
        self.assertEqual(task["elapsed_seconds"], 60)
        self.assertIsNone(task["estimate"])
        self.assertEqual(task["elapsed_label"], "已用 1 分钟")

    def test_explicit_long_task_is_visible_at_assignment(self):
        self.hooks.on_task_start("a", "重构鉴权", long_task=True)
        task = self.hooks.snapshot().to_dict()["tasks"][0]
        self.assertEqual(task["title"], "重构鉴权")
        self.assertEqual(task["elapsed_seconds"], 0)

    def test_finish_returns_actual_duration_and_clears_widget(self):
        self.hooks.on_task_start("a", "重构鉴权", long_task=True)
        self.clock.advance(12.5)
        result = self.hooks.on_task_finish("a")
        self.assertEqual(result.total_wall, 12.5)
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.hooks.snapshot().to_dict()["tasks"], [])
        self.assertIsNone(self.hooks.on_task_finish("a"))

    def test_short_task_never_leaves_a_visible_widget(self):
        updates = []
        self.hooks.subscribe(lambda state: updates.append(state.to_dict()))
        self.hooks.on_task_start("a", "改错别字")
        self.clock.advance(3)
        self.hooks.refresh()
        self.hooks.on_task_finish("a")
        self.assertTrue(all(not state["tasks"] for state in updates))

    def test_concurrent_tasks_finish_and_cancel_independently(self):
        self.hooks.on_task_start("a", "任务 A", long_task=True)
        self.clock.advance(10)
        self.hooks.on_task_start("b", "任务 B", long_task=True)
        cancelled = self.hooks.on_task_cancel("a")
        self.clock.advance(5)
        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(cancelled.total_wall, 10)
        remaining = self.hooks.snapshot().to_dict()["tasks"]
        self.assertEqual([task["task_id"] for task in remaining], ["b"])
        self.assertEqual(remaining[0]["elapsed_seconds"], 5)

    def test_failed_task_is_removed(self):
        self.hooks.on_task_start("a", "任务", long_task=True)
        self.assertEqual(self.hooks.on_task_fail("a").status, "failed")
        self.assertEqual(self.hooks.snapshot().to_dict()["tasks"], [])

    def test_duplicate_start_does_not_reset_clock(self):
        self.hooks.on_task_start("a", "原始任务", long_task=True)
        self.clock.advance(20)
        self.hooks.on_task_start("a", "重复事件", long_task=True)
        self.assertEqual(self.hooks.on_task_finish("a").total_wall, 20)
        with self.assertRaises(ValueError):
            self.hooks.on_task_start("a", "复用旧 ID")

    def test_custom_threshold_and_short_hint_still_allow_elapsed_fallback(self):
        with ElapsedHooks(prior_table(), clock=self.clock,
                          policy=DisplayPolicy(long_task_seconds=5)) as hooks:
            hooks.on_task_start("a", "原本很短", long_task=False)
            self.clock.advance(5)
            self.assertEqual(len(hooks.refresh().tasks), 1)

    def test_unsubscribe_stops_delivery_and_close_clears_state(self):
        updates = []
        unsubscribe = self.hooks.subscribe(lambda state: updates.append(state.to_dict()))
        self.hooks.on_task_start("a", "任务", long_task=True)
        unsubscribe()
        before = len(updates)
        self.clock.advance(5)
        self.hooks.refresh()
        self.assertEqual(len(updates), before)
        self.hooks.close()
        self.assertEqual(self.hooks.snapshot().to_dict()["tasks"], [])
        with self.assertRaises(RuntimeError):
            self.hooks.on_task_start("b", "已关闭")

    def test_reentrant_finish_never_publishes_old_task_to_later_subscribers(self):
        updates = []

        def finish_when_visible(state):
            if state.tasks:
                self.hooks.on_task_finish("a")

        self.hooks.subscribe(finish_when_visible)
        self.hooks.subscribe(lambda state: updates.append(state.to_dict()["tasks"]))
        self.hooks.on_task_start("a", "同步结束的任务")
        self.clock.advance(60)
        self.assertEqual(self.hooks.refresh().tasks, ())
        self.assertEqual(self.hooks.snapshot().tasks, ())
        self.assertTrue(all(not tasks for tasks in updates), updates)

    def test_threshold_rejects_nonpositive_and_nonfinite_values(self):
        for value in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DisplayPolicy(long_task_seconds=value)


class BackgroundEstimateTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()

    def hooks(self, result, **kwargs):
        hooks = ElapsedHooks(prior_table(), clock=self.clock,
                             estimator=lambda *args, **kw: result, **kwargs)
        self.addCleanup(hooks.close)
        return hooks

    def test_calibrated_prediction_can_show_a_long_task_early(self):
        hooks = self.hooks(prediction())
        hooks.on_task_start("a", "任务", g=G).result(timeout=3)
        task = hooks.snapshot().to_dict()["tasks"][0]
        self.assertEqual(task["band"], format_expectation(120, 180))
        self.assertEqual(task["estimate"]["seed"], 7)
        self.assertFalse(task["over_estimate"])
        self.clock.advance(181)
        self.assertTrue(hooks.refresh().to_dict()["tasks"][0]["over_estimate"])

    def test_prior_prediction_does_not_make_a_short_task_visible(self):
        hooks = self.hooks(prediction(calibrating=True))
        hooks.on_task_start("a", "任务", g=G).result(timeout=3)
        self.assertEqual(hooks.snapshot().tasks, ())
        self.clock.advance(60)
        task = hooks.refresh().to_dict()["tasks"][0]
        self.assertTrue(task["estimate"]["calibrating"])

    def test_uncalibrated_prediction_never_claims_a_duration_band_or_overrun(self):
        hooks = self.hooks(prediction(calibrating=True))
        hooks.on_task_start("a", "任务", g=G, long_task=True).result(timeout=3)
        self.clock.advance(600)
        task = hooks.refresh().to_dict()["tasks"][0]
        self.assertIsNone(task["band"])
        self.assertFalse(task["over_estimate"])
        self.assertEqual(task["elapsed_seconds"], 600)
        self.assertIsNotNone(task["estimate"], "先验仅保留供诊断，不作为已验证预估显示")

    def test_short_prediction_and_explicit_short_hint_stay_hidden(self):
        hooks = self.hooks(prediction(p50=20, p90=90))
        hooks.on_task_start("a", "小任务", g=G).result(timeout=3)
        self.assertEqual(hooks.snapshot().tasks, ())
        forced_short = self.hooks(prediction())
        forced_short.on_task_start("b", "小任务", g=G, long_task=False).result(timeout=3)
        self.assertEqual(forced_short.snapshot().tasks, ())

    def test_duplicate_start_keeps_same_estimate_and_future(self):
        hooks = self.hooks(prediction())
        first = hooks.on_task_start("a", "任务", g=G)
        first.result(timeout=3)
        second = hooks.on_task_start("a", "重复", g=G)
        self.assertIs(first, second)
        self.clock.advance(5)
        outcome = hooks.on_task_finish("a")
        self.assertEqual(outcome.estimate.fit_id, "fixture")
        self.assertEqual(outcome.total_wall, 5)

    def test_queued_estimate_keeps_assignment_features_and_probabilities(self):
        entered, release = threading.Event(), threading.Event()
        seen = []

        def estimator(g, probs, **kwargs):
            if not seen:
                seen.append(None)
                entered.set()
                release.wait(3)
            else:
                seen.append((dict(g), [list(pair) for pair in probs]))
            return prediction()

        with ElapsedHooks(prior_table(), clock=self.clock, estimator=estimator) as hooks:
            hooks.on_task_start("blocker", "占用估算线程", g=G)
            try:
                self.assertTrue(entered.wait(1))
                features = dict(G)
                probs = [["patch", .7], ["debug", .3]]
                future = hooks.on_task_start("queued", "排队的任务", g=features, probs=probs)
                features["task_class"] = "question"
                probs[0][:] = ["question", 1]
                probs[1][1] = 0
                release.set()
                future.result(timeout=3)
                self.assertEqual(seen[1], (G, [["patch", .7], ["debug", .3]]))
            finally:
                release.set()

    def test_close_from_host_subscriber_cannot_deadlock_with_worker_publication(self):
        # Bound the child process so a lock/join regression cannot hang the suite.
        code = textwrap.dedent('''
            import threading
            import time
            from agent_elapsed.constants import prior_table
            from agent_elapsed.hooks import ElapsedHooks
            from tests.test_hooks import G, prediction

            release = threading.Event()
            returned = []
            def estimator(*args, **kwargs):
                if not release.wait(2):
                    raise AssertionError('release not received')
                return prediction()

            hooks = ElapsedHooks(prior_table(), estimator=estimator)
            def subscriber(state):
                if state.tasks and state.tasks[0].estimate is None:
                    release.set()
                    deadline = time.monotonic() + 2
                    while hooks.snapshot().tasks[0].estimate is None:
                        if time.monotonic() > deadline:
                            raise AssertionError('estimate not stored')
                        time.sleep(.001)
                    hooks.close()
                    returned.append(True)

            hooks.subscribe(subscriber)
            future = hooks.on_task_start('a', 'A', g=G, long_task=True)
            future.result(timeout=2)
            assert returned == [True], returned
            assert not hooks.snapshot().tasks
        ''')
        result = subprocess.run([sys.executable, '-c', code],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_close_from_worker_subscriber_does_not_join_current_thread(self):
        release = threading.Event()
        returned = []

        def estimator(*args, **kwargs):
            release.wait(3)
            return prediction()

        with ElapsedHooks(prior_table(), clock=self.clock, estimator=estimator) as hooks:
            def subscriber(state):
                if state.tasks and state.tasks[0].estimate is not None:
                    hooks.close()
                    returned.append(True)

            hooks.subscribe(subscriber)
            future = hooks.on_task_start('a', '任务', g=G, long_task=True)
            release.set()
            future.result(timeout=3)
            self.assertEqual(returned, [True])
            self.assertEqual(hooks.snapshot().tasks, ())

    def test_slow_estimator_does_not_block_and_late_result_cannot_revive_task(self):
        release, entered, returned = threading.Event(), threading.Event(), threading.Event()
        futures = []

        def slow(*args, **kwargs):
            entered.set()
            release.wait(3)
            return prediction()

        with ElapsedHooks(prior_table(), clock=self.clock, estimator=slow) as hooks:
            worker = threading.Thread(target=lambda: (
                futures.append(hooks.on_task_start("a", "任务", g=G, long_task=True)), returned.set()))
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                self.assertTrue(returned.wait(1), "任务开始 hook 被预估计算阻塞")
                hooks.on_task_cancel("a")
                release.set()
                worker.join(2)
                futures[0].result(timeout=2)
                self.assertEqual(hooks.snapshot().tasks, ())
            finally:
                release.set()
                worker.join(3)

    def test_estimator_error_keeps_task_alive_and_records_diagnostic(self):
        def broken(*args, **kwargs):
            raise ValueError("bad features")

        with ElapsedHooks(prior_table(), clock=self.clock, estimator=broken) as hooks:
            hooks.on_task_start("a", "任务", g=G).result(timeout=3)
            self.clock.advance(60)
            self.assertEqual(len(hooks.refresh().tasks), 1)
            outcome = hooks.on_task_finish("a")
            self.assertIsNone(outcome.estimate)
            self.assertIn("bad features", outcome.estimation_error)

    def test_default_estimator_uses_existing_math_entry(self):
        with ElapsedHooks(prior_table(), clock=self.clock) as hooks:
            hooks.on_task_start("a", "任务", g=G, long_task=True).result(timeout=3)
            outcome = hooks.on_task_finish("a")
            self.assertEqual(outcome.estimate.n_paths, 5000)
            self.assertTrue(outcome.estimate.calibrating)
            self.assertGreater(outcome.estimate.p90, outcome.estimate.p50)
            self.assertGreaterEqual(outcome.estimator_wall, 0)


if __name__ == "__main__":
    unittest.main()
