import unittest

import numpy as np

from agent_elapsed.constants import prior_table
from agent_elapsed.schema import parse_task_row
from offline.drift import (
    detect, discount_weights, ewma_line, per_task_pinball,
)
from offline.synth import generate_tasks, truth_table


def records(rows):
    out = []
    for r in rows:
        rec, _ = parse_task_row(r)
        out.append(rec)
    return out


class DetectorTest(unittest.TestCase):
    def test_stable_stream_does_not_alert(self):
        tasks = records(generate_tasks(120, seed=51)[1])
        scores = per_task_pinball(truth_table(), tasks, m=800, seed_base=3)
        self.assertEqual(len(scores), len(tasks))
        self.assertFalse(detect(scores, baseline_n=30)["alert"])

    def test_injected_drift_alerts_within_finite_steps(self):
        tasks = records(generate_tasks(120, seed=52)[1])
        scores = np.array(per_task_pinball(truth_table(), tasks, m=800, seed_base=3))
        scores[60:] *= 3.0                       # 模拟换模型后的整体劣化
        res = detect(list(scores), baseline_n=30, k=2.0)
        self.assertTrue(res["alert"])
        self.assertIsNotNone(res["index"])
        self.assertLessEqual(res["index"] - 60, 12)

    def test_baseline_itself_is_never_an_alert(self):
        scores = [1.0] * 25 + [50.0] * 5
        self.assertFalse(detect(scores, baseline_n=30)["alert"])

    def test_ewma_upper_tracks_level(self):
        line = ewma_line([float(x) for x in np.ones(40)], alpha=0.3)
        self.assertEqual(len(line), 40)
        self.assertGreaterEqual(line[-1]["upper"], line[-1]["ewma"])

    def test_discount_weights_are_recent_heavy_and_normalised(self):
        w = discount_weights(100, half_life_tasks=20)
        self.assertEqual(len(w), 100)
        self.assertLess(w[0], w[-1])
        self.assertAlmostEqual(float(w.mean()), 1.0, places=6)
        self.assertAlmostEqual(float(w[-1] / w[-21]), 2.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
