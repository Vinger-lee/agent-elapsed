import tempfile
import dataclasses
import unittest
from pathlib import Path

from agent_elapsed.constants import dump_tables, load_tables, prior_table
from agent_elapsed.store import TelemetryStore
from offline.cli import main, run_refit
from offline.synth import generate_tasks


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
