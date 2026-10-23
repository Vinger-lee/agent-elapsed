"""常数表的形状、按环境选取、以及 calibrating 的唯一判据（spec §6.5）。"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from agent_elapsed.features import FEATURE_COLUMNS, G_SCHEMA_VERSION, N_FEATURES

OPS: tuple[str, ...] = ("llm", "tool", "build", "net")

MIN_STEP_OBS = 200
MIN_TASKS_KS = 40
MIN_NEW_OBS_PER_OP = 50
MIN_TOTAL_TASKS = 80
MIN_FIT_TASKS = 60
MC_PATHS = 5000
ETA_CLIP = 20.0
# φ（离散度）的合法区间：1 = 步长噪声已解释全部残差；25 = 上限，防止把
# 一个明显错误的常数表靠无限展宽“修好”。
PHI_MIN = 1.0
PHI_MAX = 25.0

PRIOR_MEDIAN_SECONDS = {"llm": 4.0, "tool": 1.5, "build": 90.0, "net": 3.0}
PRIOR_SIGMA = 0.8
# 先验的"等效观测数"。冷启动的这两个数字都只是猜测，权重必须小到让数据说话：
# * 位置（中位数）：1 个伪观测。给 20 个意味着 285 个真实观测之后中位数仍被拉高 33%，
#   那是系统性偏差，会让覆盖率冲到 1.0、验收门永远拒绝。
# * 尺度（log 标准差）：2 个伪观测。给 20 个时，步长本来很紧（例如 σ=0.02）的宿主需要
#   三万步以上才把 σ̂ 从 0.2 压下来，期间区间一直过宽、门也一直拒绝。
LOCATION_PRIOR_N = 1.0
PRIOR_EQUIV_N = 2.0
PRIOR_LAMBDA0 = {"llm": 8.0, "tool": 12.0, "build": 1.5, "net": 2.0}


class EnvError(KeyError):
    pass


class SchemaMismatch(ValueError):
    pass


@dataclass(frozen=True)
class StepParams:
    mu: float
    sigma: float
    step_obs: int


@dataclass(frozen=True)
class ConstTable:
    env: str
    g_schema_version: int
    fit_id: str
    trained_through_ts: float
    phi: float
    task_count: int
    ks_passed: bool
    intercept: dict[str, float]
    coef: dict[str, tuple[float, ...]]
    step: dict[str, StepParams]

    def sigma_used(self, op: str) -> float:
        """Effective log-scale standard deviation σ·√max(φ,1)."""
        return math.sqrt(self.step[op].sigma ** 2 * max(self.phi, 1.0))

    def log_params(self, op: str) -> tuple[float, float]:
        """(μ_eff, σ_eff) for the mean-preserving dispersion φ.

        σ_eff² = σ²·max(φ,1) and μ_eff = μ - (σ_eff²-σ²)/2, so the lognormal
        mean exp(μ+σ²/2) is untouched by φ while E[X²] grows with σ_eff².
        Sampling and closed-form moments must both go through here; otherwise
        the φ-widened σ would be paired with the un-shifted μ and φ would leak
        into the mean (the defect fixed in docs/math-upgrade.md).
        """
        base = self.step[op]
        sigma2 = base.sigma ** 2
        eff2 = sigma2 * max(self.phi, 1.0)
        return base.mu - (eff2 - sigma2) / 2.0, math.sqrt(eff2)

    def is_calibrating(self) -> bool:
        return not (
            all(p.step_obs >= MIN_STEP_OBS for p in self.step.values())
            and self.task_count >= MIN_TASKS_KS
            and self.ks_passed
        )


def prior_table(env: str = "default") -> ConstTable:
    return ConstTable(
        env=env, g_schema_version=G_SCHEMA_VERSION, fit_id=f"prior-{env}",
        trained_through_ts=0.0, phi=1.0, task_count=0, ks_passed=False,
        intercept={o: math.log(PRIOR_LAMBDA0[o]) for o in OPS},
        coef={o: tuple(0.0 for _ in range(N_FEATURES)) for o in OPS},
        step={o: StepParams(mu=math.log(PRIOR_MEDIAN_SECONDS[o]),
                            sigma=PRIOR_SIGMA, step_obs=0) for o in OPS},
    )


def table_to_json(t: ConstTable) -> dict:
    return {
        "fit_id": t.fit_id, "trained_through_ts": t.trained_through_ts,
        "phi": t.phi, "task_count": t.task_count, "ks_passed": t.ks_passed,
        "intercept": {o: t.intercept[o] for o in OPS},
        "coef": {o: list(t.coef[o]) for o in OPS},
        "step": {o: {"mu": t.step[o].mu, "sigma": t.step[o].sigma,
                     "step_obs": t.step[o].step_obs} for o in OPS},
    }


def table_from_json(env: str, g_schema_version: int, payload: dict) -> ConstTable:
    table = ConstTable(
        env=env, g_schema_version=g_schema_version, fit_id=payload["fit_id"],
        trained_through_ts=float(payload["trained_through_ts"]),
        phi=float(payload["phi"]), task_count=int(payload["task_count"]),
        ks_passed=bool(payload["ks_passed"]),
        intercept={o: float(payload["intercept"][o]) for o in OPS},
        coef={o: tuple(float(x) for x in payload["coef"][o]) for o in OPS},
        step={o: StepParams(mu=float(payload["step"][o]["mu"]),
                            sigma=float(payload["step"][o]["sigma"]),
                            step_obs=int(payload["step"][o]["step_obs"])) for o in OPS},
    )
    _validate_table(table)
    return table


def _validate_table(table: ConstTable) -> None:
    if table.g_schema_version != G_SCHEMA_VERSION:
        raise SchemaMismatch('常数表与当前特征版本不一致')
    if not math.isfinite(table.phi):
        raise ValueError('phi 必须有限')
    for op in OPS:
        if len(table.coef[op]) != N_FEATURES:
            raise SchemaMismatch('常数系数宽度必须为 21')
        values = (table.intercept[op], *table.coef[op], table.step[op].mu, table.step[op].sigma)
        if not all(math.isfinite(v) for v in values) or table.step[op].sigma < 0:
            raise ValueError('常数参数必须有限，sigma 不得为负')


def dump_tables(tables: dict[str, ConstTable], path: Path, backup: bool = True) -> None:
    """唯一的常数表写入口：临时文件 + os.replace，永不留下半截文件。
    覆盖前把旧版存成 constants.prev.json，供 gate.rollback 使用（spec §6.2 第 4 条）。"""
    import os

    for table in tables.values():
        _validate_table(table)
    doc = {"g_schema_version": G_SCHEMA_VERSION, 'prior_equiv_n':PRIOR_EQUIV_N,
           "columns": list(FEATURE_COLUMNS),
           "ops": list(OPS), "envs": {k: table_to_json(v) for k, v in tables.items()}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True),
                   encoding="utf-8")
    if backup and path.exists():
        path.with_suffix(".prev.json").write_text(
            path.read_text(encoding="utf-8"), encoding="utf-8")
    os.replace(tmp, path)


def bundled_prior_tables() -> dict[str, ConstTable]:
    """The cold-start table shipped inside the package.

    `pip install agent-elapsed` 之后仓库根的 constants.json 并不存在，宿主需要一个
    不依赖检出目录的起点；`offline.cli init` 就是把它写到用户自己的路径上。
    """
    from importlib import resources

    text = resources.files("agent_elapsed").joinpath(
        "data/constants.json").read_text(encoding="utf-8")
    return _tables_from_doc(json.loads(text))


def load_tables(path) -> dict[str, ConstTable]:
    return _tables_from_doc(json.loads(Path(path).read_text(encoding="utf-8")))


def _tables_from_doc(doc: dict) -> dict[str, ConstTable]:
    version = int(doc["g_schema_version"])
    if version != G_SCHEMA_VERSION:
        raise SchemaMismatch(
            f"常数表是 g_schema_version={version}，本代码只认 {G_SCHEMA_VERSION}；"
            f"跨版本混用会让权重静默失真（spec §4.3）")
    if list(doc.get("columns", FEATURE_COLUMNS)) != list(FEATURE_COLUMNS):
        raise SchemaMismatch("常数表的列顺序与当前编码不一致")
    if tuple(doc.get("ops", OPS)) != OPS:
        raise SchemaMismatch("常数表的操作类型顺序与当前 OPS 不一致")
    return {env: table_from_json(env, version, payload)
            for env, payload in doc["envs"].items()}


def pick_env(tables: dict[str, ConstTable], env_key: str) -> ConstTable:
    if env_key in tables:
        return tables[env_key]
    if "default" in tables:
        return tables["default"]
    raise EnvError(f"没有环境 {env_key!r}，也没有 default 分组可回退")
