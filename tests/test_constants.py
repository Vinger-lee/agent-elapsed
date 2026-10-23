import json
import math
import tempfile
import unittest
from pathlib import Path

from agent_elapsed.constants import (
    ConstTable, EnvError, MIN_STEP_OBS, MIN_TASKS_KS, OPS, SchemaMismatch,
    dump_tables, load_tables, pick_env, prior_table,
)
from agent_elapsed.features import N_FEATURES


def expected_from(t):
    return sum(math.exp(t.intercept[o])
               * math.exp(t.step[o].mu + t.step[o].sigma ** 2 / 2.0) for o in OPS)


class PriorTableTest(unittest.TestCase):
    def test_cold_start_expected_total_is_the_documented_magnitude(self):
        """spec §3.5 的两个口径不能混：Σ λ0·中位数 = 191 秒（文字量级），
        对数正体的真实 E[T] 还要乘 e^{σ²/2}。测试同时守这两条。"""
        t = prior_table()
        median_based = sum(math.exp(t.intercept[o]) * math.exp(t.step[o].mu) for o in OPS)
        mean_based = expected_from(t)
        self.assertAlmostEqual(median_based, 191.0, delta=1.0)
        self.assertAlmostEqual(mean_based / median_based, math.exp(0.8 ** 2 / 2), places=6)

    def test_prior_has_zero_coefficients_and_phi_one(self):
        t = prior_table()
        self.assertEqual(set(t.coef) , set(OPS))
        self.assertEqual(len(t.coef["llm"]), N_FEATURES)
        self.assertEqual(sum(t.coef["llm"]), 0.0)
        self.assertEqual(t.phi, 1.0)
        self.assertTrue(t.is_calibrating())


class CalibratingFlagTest(unittest.TestCase):
    def test_all_conditions_required(self):
        t = prior_table()
        warmed = ConstTable(
            env=t.env, g_schema_version=t.g_schema_version, fit_id="f",
            trained_through_ts=1.0, phi=t.phi, task_count=MIN_TASKS_KS, ks_passed=True,
            intercept=t.intercept, coef=t.coef,
            step={o: type(t.step[o])(t.step[o].mu, t.step[o].sigma, MIN_STEP_OBS)
                  for o in OPS})
        self.assertFalse(warmed.is_calibrating())
        self.assertTrue(ConstTable(
            env=t.env, g_schema_version=1, fit_id="f", trained_through_ts=1.0, phi=1.0,
            task_count=MIN_TASKS_KS, ks_passed=False, intercept=t.intercept, coef=t.coef,
            step=t.step).is_calibrating())

    def test_one_thin_op_holds_the_whole_flag(self):
        t = prior_table()
        counts = {o: MIN_STEP_OBS for o in OPS}
        counts["build"] = MIN_STEP_OBS - 1
        step = {o: type(t.step[o])(t.step[o].mu, t.step[o].sigma, counts[o]) for o in OPS}
        self.assertTrue(ConstTable(
            env="default", g_schema_version=1, fit_id="f", trained_through_ts=1.0, phi=1.0,
            task_count=500, ks_passed=True, intercept=t.intercept, coef=t.coef,
            step=step).is_calibrating())


class PhiWideningTest(unittest.TestCase):
    def test_sigma_scales_with_sqrt_phi(self):
        t = prior_table()
        self.assertAlmostEqual(t.sigma_used("llm"), t.step["llm"].sigma)
        wide = ConstTable(
            env=t.env, g_schema_version=1, fit_id="f", trained_through_ts=0.0, phi=4.0,
            task_count=0, ks_passed=False, intercept=t.intercept, coef=t.coef, step=t.step)
        self.assertAlmostEqual(wide.sigma_used("llm"), t.step["llm"].sigma * 2.0)

    def test_phi_below_one_is_clamped(self):
        t = prior_table()
        shrunk = ConstTable(
            env=t.env, g_schema_version=1, fit_id="f", trained_through_ts=0.0, phi=0.2,
            task_count=0, ks_passed=False, intercept=t.intercept, coef=t.coef, step=t.step)
        self.assertAlmostEqual(shrunk.sigma_used("llm"), t.step["llm"].sigma)


class RoundTripAndEnvTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "constants.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_dump_load_round_trip(self):
        tables = {"default": prior_table(), "m-a|h-a": prior_table("m-a|h-a")}
        dump_tables(tables, self.path)
        back = load_tables(self.path)
        self.assertEqual(set(back), set(tables))
        self.assertEqual(back["m-a|h-a"].intercept, tables["m-a|h-a"].intercept)
        self.assertEqual(back["default"].step["build"].mu,
                         tables["default"].step["build"].mu)

    def test_dump_leaves_a_rollback_copy_on_overwrite(self):
        dump_tables({"default": prior_table()}, self.path)
        dump_tables({"default": prior_table("default")}, self.path)
        self.assertTrue(self.path.with_suffix(".prev.json").exists())

    def test_schema_version_mismatch_raises_on_load(self):
        dump_tables({"default": prior_table()}, self.path)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        raw["g_schema_version"] = 99
        self.path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(SchemaMismatch):
            load_tables(self.path)

    def test_pick_env_falls_back_to_default(self):
        tables = {"default": prior_table()}
        self.assertEqual(pick_env(tables, "nope|nowhere").env, "default")

    def test_pick_env_without_default_raises(self):
        with self.assertRaises(EnvError):
            pick_env({"m-a|h-a": prior_table("m-a|h-a")}, "m-b|h-b")
