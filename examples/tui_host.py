"""A host owns its whole terminal frame; the widget only returns layout text."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_elapsed.bridge import handle_event
from agent_elapsed.constants import load_tables, pick_env
from agent_elapsed.hooks import ElapsedHooks
from agent_elapsed.tui import render_tui

G = {"task_class": "patch", "touched_files": "2-3", "verification_loops": "2",
     "needs_network": False, "plan_steps": "4-8"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--plain", action="store_true", help="print frames without terminal control codes")
    parser.add_argument("--cancel", action="store_true", help="cancel the long demo task instead of completing it")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("interval must be finite and positive")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    history = ["你 > 你好", "Agent > 你好，可以给我安排一个任务。"]
    with ElapsedHooks(pick_env(load_tables(ROOT / "constants.json"), "default")) as hooks:
        def paint():
            body = ["AGENT 工作区 / TUI 接入示例", "", *history]
            widget = render_tui(hooks.refresh(), width=72)
            if widget:
                body.extend(("", widget))
            body.extend(("", "────────────────────────────────────", "输入区 >"))
            if sys.stdout.isatty() and not args.plain:
                print("\x1b[2J\x1b[H", end="")
            print("\n".join(body), flush=True)

        handle_event(hooks, {"event": "prompt_received", "intent": "chat"})
        paint()
        time.sleep(args.interval)
        history.extend(("你 > 修正一处拼写", "Agent > 正在修改。"))
        hooks.on_task_start("short", "修正一处拼写", long_task=False)
        paint()
        time.sleep(args.interval)
        hooks.on_task_finish("short")
        history.append("Agent > 拼写已修正。")
        history.extend(("你 > 修复接口并运行测试", "Agent > 开始处理。"))
        hooks.on_task_start("long", "修复接口并运行测试", g=G, long_task=True)
        for _ in range(3):
            time.sleep(args.interval)
            paint()
        outcome = (hooks.on_task_cancel("long") if args.cancel else hooks.on_task_finish("long"))
        history.append("Agent > 任务已取消。" if args.cancel else "Agent > 任务已完成。")
        paint()
        print(f"宿主记录：{outcome.status}，实际 {int(outcome.total_wall)} 秒。")


if __name__ == "__main__":
    main()
