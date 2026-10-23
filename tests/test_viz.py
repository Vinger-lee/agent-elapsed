import re
import unittest

from agent_elapsed.constants import StepParams, ConstTable, OPS, prior_table
from agent_elapsed.estimate import estimate_task
from agent_elapsed.features import G_SCHEMA_VERSION
from offline.viz import badge, build_payload, format_band, inject, render_report

G = {"task_class": "patch", "touched_files": "2-3", "verification_loops": "2",
     "needs_network": False, "plan_steps": "4-8"}


def est():
    r = estimate_task(G, [("patch", 0.65), ("debug", 0.35)], table=prior_table(),
                      m=2000, seed=7)
    return r.to_dict()


class BandFormatTest(unittest.TestCase):
    """spec §7：永远是一根带，不是一个点。"""

    def test_minutes_are_integers(self):
        self.assertEqual(format_band(380.0, 840.0), "大约 6–14 分钟")

    def test_short_task_still_a_band_not_a_number(self):
        self.assertEqual(format_band(20.0, 45.0), "大约 20–45 秒")

    def test_collapsed_band_is_never_a_single_number(self):
        out = format_band(385.0, 400.0)
        self.assertIn("–", out)
        lo, hi = re.findall(r"\d+", out)
        self.assertLess(int(lo), int(hi))

    def test_no_two_decimals_anywhere(self):
        for pair in [(380.123, 840.456), (7.5, 9.25), (1.0, 59.9), (0.4, 0.6)]:
            self.assertNotRegex(format_band(*pair), r"\d\.\d\d")

    def test_boundary_60s_switches_unit(self):
        self.assertIn("秒", format_band(10.0, 59.0))
        self.assertIn("分钟", format_band(10.0, 61.0))

    def test_degenerate_band_is_still_a_band(self):
        out = format_band(100.0, 100.0)
        self.assertIn("–", out)
        lo, hi = re.findall(r"\d+", out)
        self.assertLess(int(lo), int(hi))


class BadgeTest(unittest.TestCase):
    def test_cold_start_says_calibrating(self):
        self.assertEqual(badge(prior_table()), "校准中（数据还不够）")

    def test_calibrated_shows_task_count(self):
        t = prior_table()
        warmed = ConstTable(
            env=t.env, g_schema_version=1, fit_id="f", trained_through_ts=1.0, phi=1.0,
            task_count=40, ks_passed=True, intercept=t.intercept, coef=t.coef,
            step={o: StepParams(t.step[o].mu, t.step[o].sigma, 300) for o in OPS})
        self.assertEqual(badge(warmed), "已校准 · 基于 40 个任务")
        self.assertEqual(badge(warmed, drift=True), "可能过期（检出漂移）")


class PayloadTest(unittest.TestCase):
    def test_payload_carries_no_raw_samples(self):
        payload = build_payload(est(), prior_table())
        self.assertNotIn("samples", payload)
        self.assertIn("band", payload)
        self.assertEqual(payload["band"], format_band(est()["p50"], est()["p90"]))

    def test_review_view_only_when_actual_present(self):
        self.assertIsNone(build_payload(est(), prior_table(), actual=None)["review"])
        self.assertIsNotNone(build_payload(est(), prior_table(), actual=500.0)["review"]["in_band"])

    def test_forbidden_promise_wording_absent(self):
        payload = build_payload(est(), prior_table(), actual=500.0)
        blob = str(payload)
        for banned in ("将用时", "预计完成时间"):
            self.assertNotIn(banned, blob)

    def test_two_decimals_never_appear_in_rendered_html(self):
        html = inject(build_payload(est(), prior_table()), "<div>__DATA__</div>")
        self.assertNotRegex(html, r"\b\d+\.\d{2,}\b")

    def test_injection_escapes_script_close_tag(self):
        html = inject({"x": "</script><script>alert(1)</script>"},
                      "<script>var D = __DATA__;</script>")
        self.assertNotIn("</script><script>alert", html)
        self.assertIn("<\\/script>", html)


class RenderTest(unittest.TestCase):
    def test_track_is_bounded_and_has_no_math_in_template(self):
        payload = build_payload(est(), prior_table(), actual=500.0)
        track = payload["track"]
        for key in ("left_pct", "width_pct", "marker_pct"):
            self.assertGreaterEqual(track[key], 0)
            self.assertLessEqual(track[key], 100)
        self.assertIsInstance(track["left_pct"], int)

    def test_template_contains_no_arithmetic_operators_on_data(self):
        import pathlib

        text = pathlib.Path("templates/report.html").read_text(encoding="utf-8")
        js = text.split("<script>", 1)[1].split("</script>", 1)[0]   # 只取脚本体
        for token in ("*", "/", "Math.", "parseFloat", ".reduce(", "parseInt"):
            self.assertNotIn(token, js, f"模板里不该有 {token} 这类计算")

    def test_template_is_self_contained(self):
        import pathlib

        text = (pathlib.Path("templates/report.html")).read_text(encoding="utf-8")
        self.assertNotIn("http://", text)
        self.assertNotIn("https://", text)
        for token in ("fetch(", "XMLHttpRequest", "d3.", "chart.js"):
            self.assertNotIn(token, text.lower())
        self.assertIn("__DATA__", text)

    def test_render_report_writes_a_file(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.html"
            self.assertEqual(render_report(prior_table(), out), 0)
            html = out.read_text(encoding="utf-8")
            self.assertIn("校准中", html)


if __name__ == "__main__":
    unittest.main()
