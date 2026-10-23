import unittest

from agent_elapsed.features import (
    FEATURE_COLUMNS, N_FEATURES, FeatureError, encode, encode_top2,
)

G = {"task_class": "patch", "touched_files": "2-3", "verification_loops": "2",
     "needs_network": False, "plan_steps": "4-8"}


class EncodeTest(unittest.TestCase):
    def test_matrix_width_is_21_and_keeps_every_level(self):
        self.assertEqual(N_FEATURES, 21)
        self.assertEqual(len(set(FEATURE_COLUMNS)), 21)
        self.assertNotIn("needs_network=false", FEATURE_COLUMNS[:0])  # 不 drop reference level
        self.assertIn("needs_network=false", FEATURE_COLUMNS)
        self.assertIn("needs_network=true", FEATURE_COLUMNS)

    def test_exactly_one_hot_per_feature_group(self):
        vec = encode(G)
        self.assertEqual(len(vec), 21)
        self.assertEqual(sum(vec), 5.0)
        for prefix in ("task_class", "touched_files", "verification_loops",
                       "needs_network", "plan_steps"):
            cols = [c for c in FEATURE_COLUMNS if c.startswith(prefix + "=")]
            self.assertEqual(sum(vec[FEATURE_COLUMNS.index(c)] for c in cols), 1.0, prefix)

    def test_bool_level_normalised(self):
        self.assertEqual(encode(G), encode({**G, "needs_network": "false"}))
        self.assertNotEqual(encode(G), encode({**G, "needs_network": True}))

    def test_unknown_level_raises(self):
        with self.assertRaises(FeatureError):
            encode({**G, "task_class": "magic"})

    def test_missing_feature_raises(self):
        broken = dict(G)
        broken.pop("plan_steps")
        with self.assertRaises(FeatureError):
            encode(broken)

    def test_empty_top2_falls_back_to_single_point(self):
        self.assertEqual(encode_top2(G, []), [(encode(G), 1.0)])
        self.assertEqual(encode_top2(G, iter([])), [(encode(G), 1.0)])

    def test_top2_mixes_over_task_class_only(self):
        out = encode_top2(G, [("patch", 0.65), ("debug", 0.35)])
        self.assertEqual(len(out), 2)
        (v1, p1), (v2, p2) = out
        self.assertAlmostEqual(p1, 0.65)
        self.assertAlmostEqual(p2, 0.35)
        self.assertEqual(v1[FEATURE_COLUMNS.index("task_class=patch")], 1.0)
        self.assertEqual(v2[FEATURE_COLUMNS.index("task_class=debug")], 1.0)
        for c in FEATURE_COLUMNS:
            if not c.startswith("task_class="):
                self.assertEqual(v1[FEATURE_COLUMNS.index(c)], v2[FEATURE_COLUMNS.index(c)])

    def test_unnormalised_probs_are_renormalised(self):
        out = encode_top2(G, [("patch", 2.0), ("debug", 2.0)])
        self.assertAlmostEqual(out[0][1], 0.5)
        self.assertAlmostEqual(out[1][1], 0.5)

    def test_large_finite_weights_do_not_overflow_their_normalizer(self):
        out = encode_top2(G, [("patch", 1e308), ("debug", 1e308)])
        self.assertEqual([weight for _, weight in out], [0.5, 0.5])

    def test_only_the_selected_two_probabilities_are_normalised(self):
        out = encode_top2(G, [('patch', 0.6), ('debug', 0.3), ('research', 0.1)])
        self.assertAlmostEqual(sum(p for _, p in out), 1.0)

    def test_invalid_probability_weights_are_rejected(self):
        for weight in (-0.1, float('nan'), float('inf')):
            with self.assertRaises(FeatureError):
                encode_top2(G, [('patch', weight), ('debug', 1.0)])


if __name__ == "__main__":
    unittest.main()
