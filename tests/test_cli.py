import tempfile
import dataclasses
import unittest
from pathlib import Path

from agent_elapsed.constants import dump_tables, load_tables, prior_table
from agent_elapsed.store import TelemetryStore
from offline.cli import main, run_init, run_refit, run_synth
from offline.synth import generate_tasks


class OnboardingCommandTest(unittest.TestCase):
    """装完就能跑通：init 写先验，synth 造数据，随后 refit/report 才有的可跑。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_init_writes_the_bundled_prior_and_refuses_to_clobber(self):
        target = self.root / "constants.json"
        self.assertEqual(run_init(target), 0)
        self.assertEqual(load_tables(target)["default"], prior_table())
        self.assertEqual(run_init(target), 1, "已存在时不应该默默覆盖")
        self.assertEqual(run_init(target, force=True), 0)

    def test_synth_then_refit_then_report_is_a_complete_pipeline(self):
        data, constants = self.root / "data", self.root / "constants.json"
        self.assertEqual(run_synth(data, tasks=400, seed=61), 0)
        self.assertEqual(len(TelemetryStore(data / "events.jsonl", data / "tasks.jsonl",
                                           data / "rejects.jsonl").read_tasks()), 400)
        self.assertEqual(main(["init", "--constants", str(constants)]), 0)
        code = main(["refit", "--store", str(data), "--constants", str(constants),
                     "--env", "m-a|h-a"])
        self.assertIn(code, (0, 1), "合成数据必须至少走到验收门，而不是数据不足")
        out = self.root / "report.html"
        self.assertEqual(main(["report", "--constants", str(constants), "--env", "m-a|h-a",
                               "--store", str(data), "--out", str(out)]), 0)
        html = out.read_text(encoding="utf-8")
        self.assertIn("const DATA = {", html)
        self.assertIn("fit-", html, "报告应引用拟合表的 fit_id")
        self.assertNotIn("http://", html, "报告必须自包含，不许引用外部资源")
        self.assertGreater(out.stat().st_size, 1000)

    def test_synth_rejects_a_nonsense_task_count(self):
        self.assertEqual(run_synth(self.root / "x", tasks=0, seed=1), 2)


class RefitCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.constants = root / "constants.json"
        self.store = TelemetryStore(root / "events.jsonl", root / "tasks.jsonl",
                                    root / "rejects.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self, n_tasks, env=("m-a", "h-a")):
        steps, tasks = generate_tasks(n_tasks, seed=61)
        for t in tasks:
            t["model_id"], t["host"] = env
        for r in steps:
            self.store.append_step(r)
        for r in tasks:
            self.store.append_task(r)

    def test_refit_refuses_when_below_the_batch_threshold(self):
        self.seed(50)
        code = run_refit(self.store, self.constants, env="m-a|h-a")
        self.assertEqual(code, 2)
        self.assertTrue(self.constants.exists())      # 保底写出先验表
        self.assertEqual(load_tables(self.constants)["m-a|h-a"].fit_id,
                         prior_table("m-a|h-a").fit_id)

    def test_refit_writes_an_accepted_candidate(self):
        self.seed(400)                      # holdout = 80，正好够 evaluate 的比较下限
        code = run_refit(self.store, self.constants, env="m-a|h-a")
        self.assertEqual(code, 0)
        table = load_tables(self.constants)["m-a|h-a"]
        self.assertTrue(table.fit_id.startswith("fit-"))
        self.assertGreaterEqual(table.phi, 1.0)
        self.assertGreaterEqual(table.task_count, 80)

    def test_rollback_subcommand_restores_previous(self):
        self.seed(400)
        self.assertEqual(run_refit(self.store, self.constants, env="m-a|h-a"), 0)
        first = load_tables(self.constants)["m-a|h-a"].fit_id
        tables = load_tables(self.constants)
        tables["m-a|h-a"] = dataclasses.replace(tables["m-a|h-a"], fit_id="manual-v2")
        dump_tables(tables, self.constants)
        self.assertEqual(main(["rollback", "--constants", str(self.constants)]), 0)
        self.assertEqual(load_tables(self.constants)["m-a|h-a"].fit_id, first)

    def test_unknown_env_writes_nothing_and_says_so(self):
        self.seed(400)
        self.assertEqual(run_refit(self.store, self.constants, env="nope|nowhere"), 3)
        self.assertFalse(self.constants.exists())

    def test_main_refit_touches_the_constants_file(self):
        self.seed(400)
        main(["refit", "--store", str(self.store.events_path.parent),
              "--constants", str(self.constants), "--env", "m-a|h-a"])
        self.assertTrue(self.constants.exists())
