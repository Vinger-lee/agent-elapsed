"""Ingest and runtime guards: bad data becomes a reject, never a crash or an OOM.

这些测试全部来自实测过的失败：坏 `total_wall` 曾能写进库、坏概率形状曾让落盘接口抛异常、
极端但"合法"的常数表曾让运行时去分配 700 GB。守卫必须留在原地，否则宿主会替我们崩。
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

from agent_elapsed.constants import OPS, bundled_prior_tables, prior_table
from agent_elapsed.model import MAX_EVENTS, MIN_PATHS, predict
from agent_elapsed.schema import parse_step_row, parse_task_row
from agent_elapsed.store import TelemetryStore
from offline.partition import partition_tasks

G = {"task_class": "patch", "touched_files": "2-3", "verification_loops": "2",
     "needs_network": False, "plan_steps": "4-8"}


def task_row(**over):
    row = {"v": 1, "task_id": "t1", "g_pred": G, "g_pred_probs": [["patch", 0.65]],
           "g_final": None, "total_wall": 41.5, "censored": False,
           "model_id": "m-a", "host": "h-a", "estimator_wall": 0.4, "g_schema_version": 1}
    row.update(over)
    return row


class TaskIngestTest(unittest.TestCase):
    def test_physically_impossible_totals_are_rejected(self):
        for bad in (-5.0, float("nan"), float("inf"), "abc", None):
            with self.subTest(total_wall=bad):
                record, reason = parse_task_row(task_row(total_wall=bad))
                self.assertIsNone(record)
                self.assertEqual(reason, "bad_total_wall")

    def test_malformed_probability_shapes_are_rejected_not_raised(self):
        for bad in ("notalist", [["patch"]], [["patch", 1.0, 2.0]], [["patch", -0.1]],
                    [["patch", float("nan")]], [["bogus", 0.5]], [[1, 2]]):
            with self.subTest(probs=bad):
                record, reason = parse_task_row(task_row(g_pred_probs=bad))
                self.assertIsNone(record, f"{bad} 不该被接受")
                self.assertTrue(reason.startswith("bad_probs"), reason)

    def test_missing_identity_fields_are_rejected(self):
        for field, reason_expected in (("task_id", "bad_task_id"), ("model_id", "bad_model_id"),
                                       ("host", "bad_host")):
            with self.subTest(field=field):
                record, reason = parse_task_row(task_row(**{field: ""}))
                self.assertIsNone(record)
                self.assertEqual(reason, reason_expected)

    def test_censored_must_be_boolean(self):
        record, reason = parse_task_row(task_row(censored="yes"))
        self.assertIsNone(record)
        self.assertEqual(reason, "bad_censored")

    def test_partial_features_still_parse(self):
        """契约允许宿主先落盘再补分类；缺列由离线侧隔离，不在这一层拒绝。"""
        record, reason = parse_task_row(task_row(g_pred={"task_class": "patch"}))
        self.assertIsNone(reason)
        self.assertIsNotNone(record)

    def test_step_rows_with_missing_fields_are_rejects_not_exceptions(self):
        for bad in ({}, {"op": "llm"}, {"op": "llm", "task_id": "t", "t_start": "x", "t_end": 2},
                    {"op": "llm", "task_id": "", "t_start": 1.0, "t_end": 2.0}):
            with self.subTest(row=bad):
                event, reason = parse_step_row(bad, set())
                self.assertIsNone(event)
                self.assertIsInstance(reason, str)

    def test_store_write_path_never_raises_on_bad_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TelemetryStore(Path(tmp) / "e.jsonl", Path(tmp) / "t.jsonl",
                                   Path(tmp) / "r.jsonl")
            self.assertFalse(store.append_task(task_row(g_pred_probs="nope")))
            rejects = (Path(tmp) / "r.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rejects), 1)
            self.assertEqual(json.loads(rejects[0])["reason"], "bad_probs")


class QuarantineTest(unittest.TestCase):
    def test_unreadable_features_are_counted_not_swallowed(self):
        good, _ = parse_task_row(task_row())
        broken, _ = parse_task_row(task_row(task_id="t2", g_pred={}))
        partition = partition_tasks([good, broken], prior_table(), complete_only=True)
        self.assertEqual([t.task_id for t in partition.usable], ["t1"])
        self.assertEqual(partition.summary()["skipped"], 1)
        self.assertIn("features:", partition.skipped[0][1])

    def test_censored_tasks_are_only_dropped_when_asked(self):
        record, _ = parse_task_row(task_row(censored=True))
        self.assertEqual(len(partition_tasks([record], prior_table(), complete_only=True).usable), 0)
        self.assertEqual(len(partition_tasks([record], prior_table()).usable), 1)


class RuntimeGuardTest(unittest.TestCase):
    def test_minimum_path_count_is_enforced(self):
        for m in (0, 1, -5, 2.5, True, "100"):
            with self.subTest(m=m), self.assertRaises(ValueError):
                predict(prior_table(), G, [], m=m, seed=1)
        self.assertIsNotNone(predict(prior_table(), G, [], m=MIN_PATHS, seed=1)["p50"])

    def test_absurd_rate_table_is_refused_before_allocating(self):
        """λ 被拟合成天文数字时，运行时要给可读错误，而不是去分配 700 GB。"""
        from dataclasses import replace
        table = replace(prior_table(), intercept={op: 20.0 for op in OPS})
        with self.assertRaises(ValueError) as caught:
            predict(table, G, [], m=5000, seed=1)
        message = str(caught.exception)
        self.assertIn("超过单次上限", message)
        self.assertIn(f"{MAX_EVENTS:,}", message)

    def test_the_guard_does_not_bite_a_legitimate_large_call(self):
        samples = predict(prior_table(), G, [], m=20_000, seed=1)["samples"]
        self.assertEqual(samples.shape, (20_000,))

    def test_empty_sample_set_raises_instead_of_returning_nan(self):
        import numpy as np
        from agent_elapsed.model import quantile_interval
        with self.assertRaises(ValueError):
            quantile_interval(np.array([]), 0.5)


class BundledAssetTest(unittest.TestCase):
    def test_bundled_prior_matches_the_checked_in_table(self):
        root = Path(__file__).resolve().parents[1]
        on_disk = json.loads((root / "constants.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(bundled_prior_tables()), ["default"])
        self.assertEqual(bundled_prior_tables()["default"], prior_table())
        self.assertEqual(on_disk["envs"]["default"]["fit_id"], "prior-default")

    def test_report_template_is_reachable_through_the_package(self):
        from offline.viz import template_text
        text = template_text()
        self.assertIn("__DATA__", text)
        self.assertGreater(len(text), 1000)


class PublicApiTest(unittest.TestCase):
    def test_hosts_can_import_everything_from_the_package_root(self):
        import agent_elapsed
        for name in agent_elapsed.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(agent_elapsed, name))
        self.assertTrue(callable(agent_elapsed.estimate_task))
        self.assertTrue(callable(agent_elapsed.format_expectation))


if __name__ == "__main__":
    unittest.main()
