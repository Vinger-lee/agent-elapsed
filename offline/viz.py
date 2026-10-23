"""呈现层。数学已经在 EstResult 里算完了，这里只做格式化 —— 不含任何统计代码。"""
from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

from agent_elapsed.constants import ConstTable, OPS
from agent_elapsed.presentation import format_band, format_expectation

# 模板随包分发：`pip install` 之后 `Path(__file__).parents[1]` 落在 site-packages，
# 原来的仓库相对路径会直接指向一个不存在的文件，report 命令必崩。
TEMPLATE_PACKAGE = "offline"
TEMPLATE_NAME = "templates/report.html"


def template_text() -> str:
    """The report template as shipped inside the package."""
    return resources.files(TEMPLATE_PACKAGE).joinpath(TEMPLATE_NAME).read_text(encoding="utf-8")
BANNED_PHRASES = ("将用时", "预计完成时间")
OP_LABELS = {"llm": "LLM 回合", "tool": "工具调用", "build": "构建/测试", "net": "网络"}


def badge(table: ConstTable, drift: bool = False) -> str:
    if drift:
        return "可能过期（检出漂移）"
    if table.is_calibrating():
        return "校准中（数据还不够）"
    return f"已校准 · 基于 {table.task_count} 个任务"


def contribution_rows(shares: dict[str, float]) -> list[tuple[str, str]]:
    return [(OP_LABELS.get(op, op), f"{round(shares.get(op, 0.0) * 100)}%") for op in OPS]


def _track(p50: float, p90: float, actual: float | None) -> dict:
    """布局用的比例在这里算完，模板只管贴上去 —— HTML 里不留任何计算。"""
    lo = min(p50, p90)
    hi = max(p50, p90)
    axis_hi = max(hi, float(actual) if actual is not None else 0.0)
    span_hi = axis_hi * 1.25 if axis_hi > 0 else 1.0
    scale = span_hi if span_hi > 0 else 1.0
    left = max(0.0, min(100.0, lo / scale * 100.0))
    width = max(2.0, min(100.0 - left, (hi - lo) / scale * 100.0))
    marker = None
    if actual is not None:
        marker = max(0.0, min(100.0, float(actual) / scale * 100.0))
    return {"left_pct": round(left), "width_pct": round(width),
            "marker_pct": None if marker is None else round(marker)}


def build_payload(est: dict, table: ConstTable, actual: float | None = None,
                  report: dict | None = None, drift: bool = False) -> dict:
    p50, p90 = float(est["p50"]), float(est["p90"])
    band = format_expectation(p50, p90)
    review = None
    if actual is not None:
        review = {"actual_seconds": round(float(actual)),
                  "in_band": p50 <= float(actual) <= p90,
                  "actual_label": (f'{round(actual)} 秒' if actual < 60
                                     else f'{round(actual / 60)} 分钟')}
    scorecard = None
    if report and report.get("n"):
        cov = report["taus"].get("0.9", {})
        scorecard = {"n": report["n"],
                     "hit": round(cov.get("coverage", 0.0) * report["n"]),
                     "target": 90, 'label':'P90 上限覆盖率'}
    return {"band": band, "badge": badge(table, drift=drift),
            "track": _track(p50, p90, actual),
            "contributions": contribution_rows(est.get("contributions", {})),
            "review": review, "scorecard": scorecard,
            'groups': report.get('groups', []) if report else [],
            'contribution_bars': [{'label':OP_LABELS[o],
                'width_pct':round(est.get('contributions',{}).get(o,0)*100)} for o in OPS],
            "fit_id": table.fit_id, "seed": est.get("seed"),
            "n_paths": est.get("n_paths"), "calibrating": bool(est.get("calibrating")),
            "note": "区间由蒙特卡洛复合泊松求和得到，短任务的宽带是诚实表达"}


def inject(data: dict, template_text: str) -> str:
    blob = json.dumps(data, ensure_ascii=False)
    blob = re.sub(r"</", r"<\\/", blob)          # 防止提前关闭 script 标签
    return template_text.replace("__DATA__", blob)


def retrospective_report(table: ConstTable, tasks, m: int = 2000, seed_base: int = 7):
    """最近 100 个完整任务的上限覆盖率与按预测分类分组的误差表。"""
    import numpy as np
    from offline.calibrate import calibration_report, pinball, samples_for_task

    recent = [t for t in tasks if not t.censored
              and t.g_schema_version == table.g_schema_version][-100:]
    report = calibration_report(table, recent, m=m, seed_base=seed_base)
    groups = {}
    for i, task in enumerate(recent):
        samples = samples_for_task(table, task, m=m, seed=seed_base+i)
        q50, q90 = np.quantile(samples, [.5,.9])
        group = groups.setdefault(task.g_pred['task_class'], {'n':0,'hit':0,'loss':0.0})
        group['n'] += 1
        group['hit'] += int(task.total_wall <= q90)
        group['loss'] += (pinball(task.total_wall,q50,.5)+pinball(task.total_wall,q90,.9))/2
    report['groups'] = [{'task_class':name,'n':g['n'],
        'coverage':f"{round(100*g['hit']/g['n'])}%",
        'pinball_seconds':round(g['loss']/g['n'])}
        for name,g in sorted(groups.items())]
    return report


def render_report(table: ConstTable, out: Path, est: dict | None = None,
                  actual: float | None = None, report: dict | None = None,
                  drift: bool = False) -> int:
    from agent_elapsed.estimate import estimate_task
    from agent_elapsed.features import FEATURE_LEVELS

    if est is None:
        sample_g = {name: levels[0] for name, levels in FEATURE_LEVELS.items()}
        est = estimate_task(sample_g, [], table=table).to_dict()
    payload = build_payload(est, table, actual=actual, report=report, drift=drift)
    html = inject(payload, template_text())
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return 0
