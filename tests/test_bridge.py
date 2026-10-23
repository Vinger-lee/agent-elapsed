import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

from agent_elapsed.bridge import handle_event, run_bridge
from agent_elapsed.constants import prior_table
from agent_elapsed.hooks import ElapsedHooks
from tests.test_hooks import Clock, G, prediction

ROOT = Path(__file__).resolve().parents[1]


def _json_rejects(payload: str) -> bool:
    """Does this interpreter's json scanner refuse the deeply nested input?"""
    try:
        json.loads(payload)
    except RecursionError:
        return True
    return False


class EventHookTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.hooks = ElapsedHooks(prior_table(), clock=self.clock)
        self.addCleanup(self.hooks.close)

    def test_chat_prompt_never_estimates_or_creates_a_task(self):
        calls = []

        def estimator(*args, **kwargs):
            calls.append(args)
            return prediction()

        with ElapsedHooks(prior_table(), clock=self.clock, estimator=estimator) as hooks:
            handle_event(hooks, {"event": "prompt_received", "intent": "chat", "g": G,
                                 "task_id": "chat", "title": "普通对话", "long_task": True})
            handle_event(hooks, {"event": "message", "content": "研究一下这个词是什么意思"})
            self.clock.advance(600)
            self.assertEqual(hooks.refresh().tasks, ())
        self.assertEqual(calls, [])

    def test_task_prompt_starts_timing_at_assignment_and_returns_outcome(self):
        handle_event(self.hooks, {"event": "prompt_received", "intent": "task",
                                 "task_id": "a", "title": "修复构建", "long_task": True})
        self.clock.advance(13)
        outcome = handle_event(self.hooks, {"event": "task_finished", "task_id": "a"})
        self.assertEqual(outcome.to_dict()["total_wall"], 13)
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(self.hooks.snapshot().tasks, ())

    def test_cancel_and_failed_events_clear_their_own_widgets(self):
        for task_id in ("a", "b"):
            handle_event(self.hooks, {"event": "task_started", "task_id": task_id,
                                     "title": task_id, "long_task": True})
        self.assertEqual(handle_event(self.hooks, {
            "event": "task_cancelled", "task_id": "a"}).status, "cancelled")
        self.assertEqual(len(self.hooks.snapshot().tasks), 1)
        self.assertEqual(handle_event(self.hooks, {
            "event": "task_failed", "task_id": "b"}).status, "failed")
        self.assertEqual(self.hooks.snapshot().tasks, ())

    def test_invalid_events_do_not_register_a_task(self):
        for event in ({"event": "task_started", "title": "missing ID"},
                      {"event": "task_started", "task_id": "a", "title": "bad", "g": []},
                      {"v": 2, "event": "task_started"},
                      {"event": "unknown"}, []):
            with self.subTest(event=event), self.assertRaises(ValueError):
                handle_event(self.hooks, event)
        self.clock.advance(600)
        self.assertEqual(self.hooks.snapshot().tasks, ())


class StdioBridgeTest(unittest.TestCase):
    def test_deep_json_reports_error_and_next_valid_task_still_finishes(self):
        deep = '{"event":"message","content":' + '[' * 2000 + '0' + ']' * 2000 + '}'
        events = [
            {"event": "task_started", "task_id": "next", "title": "下一任务", "long_task": True},
            {"event": "task_finished", "task_id": "next"},
            {"event": "shutdown"},
        ]
        output = io.StringIO()
        run_bridge(io.StringIO(deep + '\n' + '\n'.join(json.dumps(row) for row in events)),
                   output, table=prior_table())
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        # 新版 CPython 的 JSON 扫描器不再递归，深层输入可能被平静地接受；
        # 契约是“要么报 elapsed.error，要么安全忽略”，两种都不许拖垮后续任务。
        self.assertTrue(any(row["event"] == "elapsed.error" for row in rows)
                        or not _json_rejects(deep))
        outcomes = [row["task"] for row in rows if row["event"] == "elapsed.finished"]
        self.assertEqual([task["task_id"] for task in outcomes], ["next"])
        self.assertEqual(outcomes[0]["status"], "completed")
        self.assertEqual(rows[-2]["tasks"], [])

    def test_bad_line_reports_error_and_next_valid_task_still_finishes(self):
        source = io.StringIO('broken\n' + '\n'.join(json.dumps(row) for row in [
            {"event": "task_started", "task_id": "a", "title": "构建", "long_task": True},
            {"event": "task_finished", "task_id": "a"},
            {"event": "shutdown"}]) + '\n')
        output = io.StringIO()
        run_bridge(source, output, table=prior_table(), tick_seconds=.1)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertTrue(any(row["event"] == "elapsed.error" for row in rows))
        outcomes = [row["task"] for row in rows if row["event"] == "elapsed.finished"]
        self.assertEqual([task["task_id"] for task in outcomes], ["a"])
        self.assertEqual(outcomes[0]["status"], "completed")
        states = [row for row in rows if row["event"] == "elapsed.updated"]
        self.assertTrue(any(row["tasks"] for row in states))
        self.assertEqual(states[-1]["tasks"], [])

    def test_eof_clears_active_widgets(self):
        output = io.StringIO()
        run_bridge(io.StringIO(json.dumps({"event": "task_started", "task_id": "a",
                   "title": "任务", "long_task": True}) + '\n'), output, table=prior_table())
        self.assertEqual(json.loads(output.getvalue().splitlines()[-1])["tasks"], [])

    def test_module_cli_exchanges_utf8_events_with_non_python_host(self):
        events = [
            {"event": "message", "content": "你好"},
            {"event": "task_started", "task_id": "a", "title": "修复鉴权", "long_task": True},
            {"event": "task_cancelled", "task_id": "a"},
            {"event": "shutdown"},
        ]
        result = subprocess.run([sys.executable, "-m", "agent_elapsed.bridge",
                                 "--constants", str(ROOT / "constants.json")],
                                input='\n'.join(json.dumps(row, ensure_ascii=False)
                                                for row in events) + '\n',
                                capture_output=True, encoding="utf-8", cwd=ROOT, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        visible = [task for row in rows if row["event"] == "elapsed.updated" for task in row["tasks"]]
        self.assertEqual([task["title"] for task in visible], ["修复鉴权"])
        self.assertTrue(any(row["event"] == "elapsed.finished" and
                            row["task"]["status"] == "cancelled" for row in rows))


if __name__ == "__main__":
    unittest.main()
