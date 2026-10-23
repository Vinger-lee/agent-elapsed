import math
import dataclasses
import tempfile
import unittest
from pathlib import Path

from agent_elapsed.constants import (
    MIN_FIT_TASKS, MIN_TOTAL_TASKS, OPS, StepParams, ConstTable, dump_tables,
    load_tables, prior_table,
)
from agent_elapsed.schema import parse_step_row, parse_task_row
from offline.gate import (
    build_candidate, estimate_phi, evaluate, gate_eligible, holdout_size,
    rollback, temporal_split,
)
from offline.synth import HIDDEN_TRUTH, generate_tasks, truth_table

G = {"task_class": "patch", "touched_files": "2-3", "verification_loops": "2",
     "needs_network": False, "plan_steps": "4-8"}


def events(rows):
    seen, out = set(), []
    for r in rows:
        ev, reason = parse_step_row(r, seen)
        assert ev is not None, reason
        out.append(ev)
    return out


def records(rows):
    out = []
    for r in rows:
        rec, reason = parse_task_row(r)
        assert rec is not None, reason
        out.append(rec)
    return out


class SplitTest(unittest.TestCase):
    def test_holdout_is_at_least_20_or_20pct(self):
        self.assertEqual(holdout_size(100), 20)
        self.assertEqual(holdout_size(500), 100)
        self.assertEqual(holdout_size(60), 20)

    def test_split_is_temporal_not_random(self):
        tasks = records(generate_tasks(120, seed=41)[1])
        fit, holdout = temporal_split(tasks)
        self.assertEqual(len(fit) + len(holdout), 120)
        self.assertEqual(len(holdout), holdout_size(120))
        self.assertEqual(fit[-1].task_id, tasks[len(fit) - 1].task_id)
        self.assertEqual(holdout[0].task_id, tasks[len(fit)].task_id)

    def test_eligibility_thresholds_are_the_spec_numbers(self):
        self.assertTrue(gate_eligible(80, 60))
        self.assertFalse(gate_eligible(79, 59))
        self.assertFalse(gate_eligible(100, 59))
        self.assertEqual(MIN_TOTAL_TASKS, 80)
        self.assertEqual(MIN_FIT_TASKS, 60)

    def test_small_sample_refuses_the_gate_rather_than_pretending(self):
        tasks = records(generate_tasks(50, seed=42)[1])
        fit, holdout = temporal_split(tasks)
        self.assertFalse(gate_eligible(len(tasks), len(fit)))
        res = evaluate(prior_table(), dataclasses.replace(truth_table(), task_count=len(fit)), holdout)
        self.assertFalse(res.accepted)
        self.assertIn("insufficient", res.reason)

    def test_valid_twenty_task_holdout_does_not_require_eighty(self):
        tasks = [t for t in records(generate_tasks(100, seed=41)[1]) if not t.censored][-20:]
        result = evaluate(truth_table(), truth_table(), tasks)
        self.assertNotEqual(result.reason, 'insufficient-tasks')

    def test_tiny_split_has_no_negative_slice(self):
        tasks = list(range(15))
        self.assertEqual(temporal_split(tasks), ([], tasks))


class GateDecisionTest(unittest.TestCase):
    def setUp(self):
        _, rows = generate_tasks(400, seed=43)
        self.tasks = records(rows)
        self.fit, self.holdout = temporal_split(self.tasks)

    def test_truth_beats_prior_and_is_accepted(self):
        res = evaluate(prior_table(), truth_table(), self.holdout)
        self.assertTrue(res.accepted, msg=str(res.details))
        self.assertEqual(res.reason, "accepted")

    def test_overconfident_candidate_is_rejected(self):
        t = truth_table()
        tight = ConstTable(
            env=t.env, g_schema_version=1, fit_id="tight", trained_through_ts=0.0,
            phi=1.0, task_count=t.task_count, ks_passed=True, intercept=t.intercept,
            coef=t.coef,
            step={o: StepParams(t.step[o].mu, t.step[o].sigma * 0.3, t.step[o].step_obs)
                  for o in OPS})
        res = evaluate(t, tight, self.holdout)
        self.assertFalse(res.accepted)
        self.assertIn("calibration", res.reason)

    def test_identical_candidate_passes_the_non_strict_gate(self):
        t = truth_table()
        self.assertTrue(evaluate(t, t, self.holdout).accepted)

    def test_identical_candidate_fails_the_expansion_gate(self):
        """§6.3 扩容要求严格优于：一模一样的东西不许冒充进步。"""
        t = truth_table()
        res = evaluate(t, t, self.holdout, require_strict=True)
        self.assertFalse(res.accepted)
        self.assertIn("not strictly better", res.reason)

    def test_gate_is_deterministic_and_paired(self):
        """配对比较必须逐字重复：同一批任务、同一组种子，否则分数不可比。"""
        a = evaluate(prior_table(), truth_table(), self.holdout).details
        b = evaluate(prior_table(), truth_table(), self.holdout).details
        self.assertEqual(a["candidate_pinball"], b["candidate_pinball"])
        self.assertEqual(a["online_pinball"], b["online_pinball"])
        self.assertEqual(a["per_task_seeds"], b["per_task_seeds"])
        self.assertEqual(len(a["per_task_seeds"]), sum(not t.censored for t in self.holdout))


class BuildCandidateTest(unittest.TestCase):
    def test_holdout_steps_never_enter_candidate_fit(self):
        steps, rows = generate_tasks(100, seed=44)
        ev = events(steps)
        fit, _ = temporal_split(records(rows))
        fit_ids = {t.task_id for t in fit}
        a = build_candidate(ev, fit)
        b = build_candidate([s for s in ev if s.task_id in fit_ids], fit)
        self.assertEqual(a, b)

    def test_candidate_carries_fit_metadata(self):
        steps, rows = generate_tasks(200, seed=44)
        cand = build_candidate(events(steps), records(rows), env="m-a|h-a")
        self.assertTrue(cand.fit_id.startswith("fit-"))
        self.assertGreater(cand.trained_through_ts, 0.0)
        self.assertEqual(cand.env, "m-a|h-a")
        self.assertGreaterEqual(cand.phi, 1.0)
        self.assertEqual(set(cand.step), set(OPS))
        self.assertEqual(len(cand.coef["llm"]), 21)

    def test_candidate_step_medians_move_toward_hidden_truth(self):
        steps, rows = generate_tasks(300, seed=45)
        cand = build_candidate(events(steps), records(rows), env="default")
        for op in OPS:
            self.assertLess(abs(cand.step[op].mu - HIDDEN_TRUTH["step"][op][0]),
                            abs(prior_table().step[op].mu - HIDDEN_TRUTH["step"][op][0]), op)


class PhiTest(unittest.TestCase):
    def test_underdispersed_table_gets_phi_above_one(self):
        """步长真实波动比表里大 → φ 必须 >1，用来兜住独立性假设低估的方差。"""
        from offline.synth import HIDDEN_TRUTH as T

        wider = {"step": {o: (T["step"][o][0], T["step"][o][1] * 2.0) for o in OPS},
                 "intercept": T["intercept"], "coef": T["coef"]}
        _, rows = generate_tasks(150, seed=46, truth=wider)
        phi = estimate_phi(truth_table(), records(rows), m=1500, seed_base=5)
        self.assertGreater(phi, 1.0)

    def test_phi_is_clamped_to_at_least_one(self):
        _, rows = generate_tasks(150, seed=47)
        t = truth_table()
        narrow = ConstTable(
            env=t.env, g_schema_version=1, fit_id="n", trained_through_ts=0.0, phi=1.0,
            task_count=1000, ks_passed=True, intercept=t.intercept, coef=t.coef,
            step={o: StepParams(t.step[o].mu, t.step[o].sigma * 2.5, t.step[o].step_obs)
                  for o in OPS})
        self.assertEqual(estimate_phi(narrow, records(rows), m=1500, seed_base=5), 1.0)


class AtomicSwapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "constants.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_creates_backup_then_swaps(self):
        dump_tables({"default": prior_table()}, self.path)
        first = load_tables(self.path)["default"].fit_id
        dump_tables({"default": truth_table("default")}, self.path)
        self.assertEqual(load_tables(self.path)["default"].fit_id, "hidden-truth")
        self.assertTrue(self.path.with_suffix(".prev.json").exists())
        self.assertTrue(rollback(self.path))
        self.assertEqual(load_tables(self.path)["default"].fit_id, first)

    def test_rollback_without_backup_is_reported_not_crashing(self):
        self.assertFalse(rollback(self.path))

    def test_no_partial_file_is_left_behind(self):
        dump_tables({"default": prior_table()}, self.path)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
