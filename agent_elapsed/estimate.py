"""组件运行时唯一入口。纯函数：不读遥测、不拟合、不开网络。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from agent_elapsed.constants import MC_PATHS, ConstTable, SchemaMismatch
from agent_elapsed.features import G_SCHEMA_VERSION
from agent_elapsed.model import predict

DEFAULT_SEED = 20261022


@dataclass(frozen=True)
class EstResult:
    p50: float
    p90: float
    expected: float
    contributions: dict[str, float]
    lambdas: dict[str, float]
    calibrating: bool
    fit_id: str
    seed: int
    n_paths: int
    g_schema_version: int
    p50_se: float = 0.0
    p90_se: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_task(g: dict, probs, *, table: ConstTable,
                  m: int = MC_PATHS, seed: int = DEFAULT_SEED) -> EstResult:
    if table.g_schema_version != G_SCHEMA_VERSION:
        raise SchemaMismatch(
            f"常数表版本 {table.g_schema_version} 与代码 {G_SCHEMA_VERSION} 不符")
    out = predict(table, g, list(probs or []), m=m, seed=seed)
    return EstResult(
        p50=out["p50"], p90=out["p90"], expected=out["expected"],
        contributions=out["contributions"], lambdas=out["lambdas"],
        calibrating=table.is_calibrating(), fit_id=table.fit_id,
        seed=seed, n_paths=m, g_schema_version=table.g_schema_version,
        p50_se=out["p50_se"], p90_se=out["p90_se"],
    )
