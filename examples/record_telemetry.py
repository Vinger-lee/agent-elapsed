"""端到端示例：一个任务从 hook 到落盘、再到离线重拟合与报告。

这是"模型会越用越准"这句话在代码里的样子。运行：

    python3 examples/record_telemetry.py                 # 默认录 90 个合成任务到 ./demo-data
    python3 examples/record_telemetry.py --tasks 120 --out demo-data

它刻意不做真事：任务本体被替换成几段 sleep，步事件用 `recorder.step()` 打点。真实宿主只需要
把这三处换成自己的动作——开始任务时 `recorder.start(...)`，每个步骤包一层 `recorder.step(...)`，
结束时 `recorder.finish/cancel/fail(...)`——其余（写盘、删失标记、特征留存）都由组件负责。

录满 80 个完整任务后脚本会自动尝试一次 `refit`（含校验门），并把结果打印出来；不够就只报告缺口。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_elapsed.constants import bundled_prior_tables, load_tables, pick_env
from agent_elapsed.hooks import DisplayPolicy, ElapsedHooks
from agent_elapsed.store import TelemetryStore
from agent_elapsed.telemetry import TelemetryRecorder
from agent_elapsed.tui import render_tui
from offline.cli import run_init, run_refit

OPS = ("llm", "tool", "build", "net")
TASK_CLASSES = ("patch", "debug", "refactor", "research")
# 四类操作的典型步数（泊松均值）与步长中位数（秒）。示例刻意保留真实任务该有的抖动：
# 步数随机、步长对数正态抖动 —— 否则每个任务一模一样，任何带离散度的模型都会被判"过覆盖"。
SHAPE = {"llm": (4.0, 0.05), "tool": (5.0, 0.03), "build": (1.5, 0.20), "net": (1.0, 0.04)}
STEP_LOG_SIGMA = 0.45


def fake_task(recorder: TelemetryRecorder, task_id: str, index: int,
              rng: random.Random) -> None:
    task_class = TASK_CLASSES[index % len(TASK_CLASSES)]
    g = {"task_class": task_class, "touched_files": "2-3" if index % 2 else "1",
         "verification_loops": "1" if index % 3 else "0", "needs_network": index % 5 == 0,
         "plan_steps": "1-3"}
    probs = [(task_class, 0.7), (TASK_CLASSES[(index + 1) % len(TASK_CLASSES)], 0.3)]
    recorder.start(task_id, f"示例任务 {index}: {task_class}", g=g, probs=probs, long_task=True)
    for op in OPS:
        rate, median = SHAPE[op]
        # 任务类别影响步数：debug/patch 更碎，research 更多网络步
        if op == "llm" and task_class in ("debug", "patch"):
            rate *= 1.5
        if op == "net" and task_class == "research":
            rate *= 3.0
        for _ in range(_poisson(rng, rate)):
            seconds = median * math.exp(rng.gauss(0.0, STEP_LOG_SIGMA))
            with recorder.step(task_id, op) as step:
                time.sleep(min(seconds, 0.4))
                step["ok"] = rng.random() > 0.05
    # 每七个任务有一次被"取消"：它的总时长只是下界，必须记成删失
    if index and index % 7 == 0:
        recorder.cancel(task_id)
    else:
        recorder.finish(task_id)


def _poisson(rng: random.Random, rate: float) -> int:
    """Small-rate Poisson draw without numpy (the example stays dependency-light)."""
    limit, count, product = math.exp(-rate), 0, rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "demo-data"))
    parser.add_argument("--tasks", type=int, default=90)
    parser.add_argument("--model-id", default="demo-model")
    parser.add_argument("--host", default="demo-host")
    parser.add_argument("--quiet", action="store_true", help="不打印每个任务的组件状态")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    store = TelemetryStore(out / "events.jsonl", out / "tasks.jsonl", out / "rejects.jsonl")
    constants_path = out / "constants.json"
    if not constants_path.exists():
        run_init(constants_path)

    env = f"{args.model_id}|{args.host}"
    table = pick_env(load_tables(constants_path), env)
    rng = random.Random(20261023)
    with ElapsedHooks(table, policy=DisplayPolicy(long_task_seconds=2)) as hooks:
        recorder = TelemetryRecorder(hooks, store, model_id=args.model_id, host=args.host)
        for index in range(args.tasks):
            fake_task(recorder, f"demo-{index:03d}", index, rng)
            if not args.quiet and index % 30 == 0:
                print(render_tui(hooks.refresh(), width=72) or "（当前没有需要显示的长任务）")
                print()

    tasks = store.read_tasks()
    complete = [t for t in tasks if not t.censored]
    print(json.dumps({"recorded_tasks": len(tasks), "complete": len(complete),
                      "censored": len(tasks) - len(complete),
                      "steps": out.joinpath("events.jsonl").read_text(encoding="utf-8").count("\n")
                      if out.joinpath("events.jsonl").exists() else 0}, ensure_ascii=False))
    print(f"\n下一步：python3 -m offline.cli refit --store {out} "
          f"--constants {constants_path} --env '{env}'")
    if len(complete) >= 80:
        print("完整任务已够 80，尝试一次重拟合：")
        code = run_refit(store, constants_path, env)
        print(f"refit 退出码 = {code}（0 成功上线，2 数据不足，1 被验收门拒绝）")
    else:
        print(f"完整任务 {len(complete)} 个，还不到 80；再多录一些再来重拟合。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
