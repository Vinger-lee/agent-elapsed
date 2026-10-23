"""Regression coverage for spec boundaries missing from the plan examples."""
import contextlib
import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent_elapsed.constants import LOCATION_PRIOR_N, PRIOR_EQUIV_N, OPS, dump_tables, load_tables, prior_table
from agent_elapsed.schema import parse_step_row, parse_task_row
from agent_elapsed.store import TelemetryStore
from offline.cli import run_refit
from offline.fit_counts import design_rows, fit_counts
from offline.viz import build_payload, retrospective_report
from offline.synth import generate_tasks, truth_table
from offline.fit_step import fit_step_distributions
from offline.gate import rollback
from tests.test_schema import step_row, task_row
from tests.test_features import G


class TelemetryContractTest(unittest.TestCase):
    def test_nan_and_infinite_timestamps_are_not_valid_durations(self):
        for end in (float('nan'), float('inf')):
            event, reason = parse_step_row(step_row(t_end=end), set())
            self.assertIsNone(event)
            self.assertEqual(reason, 'non_monotonic')

    def test_foreign_step_schema_never_enters_fits(self):
        event, reason = parse_step_row(step_row(v=99), set())
        self.assertIsNone(event)
        self.assertEqual(reason, 'bad_g_schema_version')

    def test_task_rows_keep_wire_schema_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TelemetryStore(root/'events.jsonl',root/'tasks.jsonl',root/'rejects.jsonl')
            store.append_task(task_row())
            self.assertEqual(json.loads(store.tasks_path.read_text())['v'], 1)

    def test_offline_read_accepts_original_wire_rows_and_audits_invalid_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TelemetryStore(root/'events.jsonl',root/'tasks.jsonl',root/'rejects.jsonl')
            store.events_path.write_text(json.dumps(step_row())+'\n'+json.dumps(step_row())+'\n{broken\n',encoding='utf-8')
            store.tasks_path.write_text(json.dumps(task_row())+'\n',encoding='utf-8')
            self.assertEqual(len(store.read_steps()),1)
            self.assertEqual(len(store.read_tasks()),1)
            reasons={json.loads(line)['reason'] for line in store.rejects_path.read_text().splitlines()}
            self.assertEqual(reasons,{'duplicate','invalid_json'})


class CountContractTest(unittest.TestCase):
    def test_completed_tasks_with_zero_steps_remain_in_count_regression(self):
        a = parse_task_row(task_row(task_id='a',g_pred=G,censored=False))[0]
        b = dataclasses.replace(a,task_id='b')
        event = parse_step_row(step_row(task_id='a',op='llm'),set())[0]
        out = fit_counts([event],[a,b])
        self.assertAlmostEqual(np.exp(out['llm'][0]),.5,places=5)

    def test_empty_design_matrix_keeps_21_columns(self):
        self.assertEqual(design_rows([]).shape,(0,21))


class RefitThresholdTest(unittest.TestCase):
    def test_existing_constants_stay_byte_identical_when_no_new_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path=root/'constants.json'
            env='m-a|h-a'
            base=dataclasses.replace(prior_table(env),trained_through_ts=1000.0)
            dump_tables({env:base},path)
            before=path.read_bytes()
            store=TelemetryStore(root/'events.jsonl',root/'tasks.jsonl',root/'rejects.jsonl')
            store.append_task(task_row(g_pred=G))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(run_refit(store,path,env),2)
            self.assertEqual(path.read_bytes(),before)
            self.assertIn('llm',output.getvalue())
            self.assertIn('50',output.getvalue())
            self.assertFalse(path.with_suffix('.prev.json').exists())


class ReviewContractTest(unittest.TestCase):
    def test_retrospective_uses_last_hundred_complete_tasks_and_groups(self):
        tasks=[parse_task_row(r)[0] for r in generate_tasks(130,99)[1]]
        report=retrospective_report(truth_table(),tasks,m=300)
        self.assertEqual(report['n'],100)
        self.assertEqual(sum(g['n'] for g in report['groups']),100)
        self.assertTrue(all(isinstance(g['pinball_seconds'],int) for g in report['groups']))

    def test_actual_time_is_an_observation_not_an_invented_range(self):
        payload=build_payload({'p50':300,'p90':600},prior_table(),actual=500)
        self.assertNotIn('–',payload['review']['actual_label'])
        self.assertIn('8',payload['review']['actual_label'])

    def test_axis_includes_actual_outside_prediction(self):
        payload=build_payload({'p50':100,'p90':200},prior_table(),actual=1000)
        self.assertLess(payload['track']['marker_pct'],100)
        self.assertGreater(payload['track']['marker_pct'],
                           payload['track']['left_pct']+payload['track']['width_pct'])

    def test_scorecard_describes_upper_quantile_coverage(self):
        payload=build_payload({'p50':100,'p90':200},prior_table(),
            report={'n':100,'taus':{'0.9':{'coverage':.87}}})
        self.assertIn('P90',payload['scorecard']['label'])
        self.assertEqual(payload['scorecard']['hit'],87)


class WeightedStepFitTest(unittest.TestCase):
    def test_zero_weight_history_does_not_bias_recent_step_estimate(self):
        early=parse_step_row(step_row(task_id='old',t_end=101),set())[0]
        recent=parse_step_row(step_row(task_id='new',t_end=110),set())[0]
        weighted=fit_step_distributions([early,recent], weights=[0,1])
        expected=fit_step_distributions([recent])
        self.assertAlmostEqual(weighted['tool'].mu,expected['tool'].mu)
        self.assertAlmostEqual(weighted['tool'].sigma,expected['tool'].sigma)
        self.assertEqual(weighted['tool'].step_obs,2)


class ConstantsContractTest(unittest.TestCase):
    def test_constants_expose_prior_strength_for_auditing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'constants.json'
            dump_tables({'default':prior_table()},path)
            doc=json.loads(path.read_text())
            # 常数表要自证先验强度：尺度 2 / 位置 1（见 constants.py 的注释）。
            self.assertEqual(doc['prior_equiv_n'], PRIOR_EQUIV_N)
            self.assertEqual(doc['prior_equiv_n'], 2.0)
            self.assertEqual(LOCATION_PRIOR_N, 1.0)

    def test_malformed_coefficient_width_is_rejected_on_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'constants.json'
            dump_tables({'default':prior_table()},path)
            doc=json.loads(path.read_text())
            doc['envs']['default']['coef']['tool']=[0.0]
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                load_tables(path)

    def test_corrupt_backup_cannot_replace_valid_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'constants.json'
            dump_tables({'default':prior_table()},path)
            before=path.read_bytes()
            path.with_suffix('.prev.json').write_text('{broken')
            with self.assertRaises(ValueError):
                rollback(path)
            self.assertEqual(path.read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
