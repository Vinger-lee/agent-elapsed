import unicodedata
import unittest

from agent_elapsed.hooks import TaskView, WidgetState
from agent_elapsed.tui import render_tui
from tests.test_hooks import prediction


class TuiTest(unittest.TestCase):
    def test_hidden_widget_occupies_no_space(self):
        self.assertEqual(render_tui(WidgetState()), "")

    def test_render_includes_task_context_estimate_and_actual_elapsed(self):
        state = WidgetState((TaskView("a", "修复登录", 70, prediction()),))
        text = render_tui(state)
        self.assertIn("修复登录", text)
        self.assertIn("大约 2–3 分钟", text)
        self.assertIn("已用 1 分钟", text)
        self.assertNotIn("%", text)
        self.assertNotIn("\x1b", text)

    def test_no_estimate_is_still_a_task_widget(self):
        state = WidgetState((TaskView("a", "等待构建", 90, None),))
        text = render_tui(state)
        self.assertIn("等待构建", text)
        self.assertIn("任务进行中", text)
        self.assertIn("已用 1 分钟", text)

    def test_titles_cannot_inject_terminal_sequences_or_new_lines(self):
        title = "修复\x1b[2J\x1b]0;changed\x07\n登录\r\t\u202e"
        state = WidgetState((TaskView("a", title, 1, prediction()),))
        text = render_tui(state)
        self.assertEqual(len(text.splitlines()), 2)
        self.assertNotIn("changed", text)
        self.assertNotIn("\x1b", text)
        self.assertNotIn("\r", text)
        self.assertNotIn("\u202e", text)
        self.assertIn("登录", text)

    def test_narrow_host_layout_respects_chinese_cell_width(self):
        state = WidgetState((TaskView("a", "很长的任务标题" * 10, 70, prediction()),))
        for line in render_tui(state, width=24).splitlines():
            cells = sum(0 if unicodedata.combining(c) else
                        2 if unicodedata.east_asian_width(c) in "FW" else 1 for c in line)
            self.assertLessEqual(cells, 24)

    def test_exceeded_estimate_is_not_rendered_as_negative_remaining_time(self):
        state = WidgetState((TaskView("a", "任务", 200, prediction()),))
        text = render_tui(state)
        self.assertIn("已超过预估区间", text)
        self.assertNotIn("剩余", text)

    def test_uncalibrated_task_shows_elapsed_without_a_speculative_band(self):
        state = WidgetState((TaskView("a", "任务", 200, prediction(calibrating=True)),))
        text = render_tui(state)
        self.assertIn("任务进行中", text)
        self.assertIn("已用 3 分钟", text)
        self.assertIn("校准中", text)
        self.assertNotIn("大约", text)
        self.assertNotIn("已超过预估区间", text)


if __name__ == "__main__":
    unittest.main()
