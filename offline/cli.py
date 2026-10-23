"""离线显式命令面。重拟合不在运行时发生（spec §6）。"""
from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

from agent_elapsed.constants import (
    G_SCHEMA_VERSION, MIN_NEW_OBS_PER_OP, OPS, dump_tables, load_tables, pick_env, prior_table,
)
from agent_elapsed.store import TelemetryStore
from offline.gate import build_candidate, evaluate, gate_eligible, rollback, temporal_split
from offline.viz import render_report, retrospective_report
from offline.calibrate import calibration_report
from offline.drift import detect, discount_weights, per_task_pinball

EXIT_OK, EXIT_GATE_REJECTED, EXIT_INSUFFICIENT, EXIT_NO_ENV, EXIT_NO_BACKUP = 0, 1, 2, 3, 4


def _store(root: Path) -> TelemetryStore:
    root = Path(root)
    return TelemetryStore(root / "events.jsonl", root / "tasks.jsonl", root / "rejects.jsonl")


def run_refit(store: TelemetryStore, constants_path, env: str) -> int:
    """整批规模闸（≥80 总 / ≥60 拟合）唯一的判定处 —— 只有这里同时看得见两侧。"""
    constants_path = Path(constants_path)
    tables = (load_tables(constants_path) if constants_path.exists()
              else {"default": prior_table("default")})
    tasks = [t for t in store.read_tasks() if t.env_key() == env
             and t.g_schema_version == G_SCHEMA_VERSION]
    if not tasks:
        print(json.dumps({'reason':'unknown-environment','env':env}))
        return EXIT_NO_ENV          # 不静默为未知环境新建一个分组
    ids = {t.task_id for t in tasks}
    steps = [s for s in store.read_steps() if s.task_id in ids]
    starts = {}
    for step in steps:
        starts[step.task_id] = min(starts.get(step.task_id, step.t_start), step.t_start)
    order = {t.task_id:i for i,t in enumerate(tasks)}
    tasks.sort(key=lambda t: (starts.get(t.task_id, float('inf')), order[t.task_id]))
    complete = [t for t in tasks if not t.censored]
    fit_complete, holdout = temporal_split(complete)
    fit_ids = {t.task_id for t in fit_complete}
    cutoff = starts.get(holdout[0].task_id, float('inf')) if holdout else float('inf')
    fit = [t for t in tasks if t.task_id in fit_ids
           or (t.censored and starts.get(t.task_id, float('inf')) < cutoff)]
    fit_ids = {t.task_id for t in fit}
    fit_steps = [s for s in steps if s.task_id in fit_ids]
    base = tables.get(env) or prior_table(env)
    new_counts = Counter(s.op for s in fit_steps if s.t_end > base.trained_through_ts)
    deficits = {o:max(0,MIN_NEW_OBS_PER_OP-new_counts[o]) for o in OPS}
    if any(deficits.values()) or not gate_eligible(len(complete), len(fit_complete)):
        diagnostic = calibration_report(base, complete)
        print(json.dumps({'reason':'insufficient-data','missing_steps':deficits,
                          'n_total':len(complete),'n_fit':len(fit_complete),
                          'descriptive':diagnostic}, ensure_ascii=False))
        if not constants_path.exists():
            tables[env] = base
            dump_tables(tables, constants_path)
        return EXIT_INSUFFICIENT
    scores = per_task_pinball(base, fit_complete)
    drift = detect(scores)['alert']
    weights = discount_weights(len(fit), half_life_tasks=30 if drift else 120)
    candidate = build_candidate(fit_steps, fit, env=env, weights=weights)
    verdict = evaluate(base, candidate, holdout)
    print(json.dumps({'reason':verdict.reason,'drift':drift,'details':verdict.details},
                     ensure_ascii=False))
    if not verdict.accepted:
        if not constants_path.exists():
            tables[env] = base
            dump_tables(tables, constants_path)
        return EXIT_GATE_REJECTED
    # ConstTable 是 frozen：过门后的两个字段只能用 replace 生成新表，不能就地赋值
    tables[env] = dataclasses.replace(
        candidate, task_count=len(complete),
        ks_passed=bool(verdict.details["report"]["ks"]["pass"]))
    dump_tables(tables, constants_path)
    return EXIT_OK


def main(argv) -> int:
    parser = argparse.ArgumentParser(prog="agent-elapsed")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_refit = sub.add_parser("refit", help="离线重拟合，过验收门才上线")
    p_refit.add_argument("--store", required=True, help="存放两个 JSONL 的目录")
    p_refit.add_argument("--constants", required=True)
    p_refit.add_argument("--env", required=True, help="model_id|host")

    p_report = sub.add_parser("report", help="渲染自包含 HTML 报告")
    p_report.add_argument("--constants", required=True)
    p_report.add_argument("--env", required=True)
    p_report.add_argument("--out", required=True)
    p_report.add_argument('--store', help='可选：读取遥测，显示最近 100 个任务复盘')
    p_report.add_argument('--estimate', help='可选：任务开始时保存的 EstResult JSON')
    p_report.add_argument('--actual', type=float, help='可选：本次任务实际秒数')

    p_roll = sub.add_parser("rollback", help="退回上一版常数表")
    p_roll.add_argument("--constants", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "refit":
        return run_refit(_store(Path(args.store)), Path(args.constants), args.env)
    if args.cmd == "rollback":
        return EXIT_OK if rollback(Path(args.constants)) else EXIT_NO_BACKUP
    tables = load_tables(Path(args.constants))
    table = pick_env(tables, args.env)
    est = json.loads(Path(args.estimate).read_text(encoding='utf-8')) if args.estimate else None
    report, drift = None, False
    if args.store:
        tasks = [t for t in _store(Path(args.store)).read_tasks()
                 if t.env_key() == args.env and not t.censored]
        report = retrospective_report(table, tasks)
        drift = detect(per_task_pinball(table, tasks))['alert']
    return render_report(table, Path(args.out), est=est, actual=args.actual,
                         report=report, drift=drift)


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
