"""Agent 任务用时预估组件的运行时面：只读常数表，不含拟合、不做 I/O。

公开入口都从这里出去，宿主不必知道内部模块怎么分层：

    from agent_elapsed import (ElapsedHooks, DisplayPolicy, TelemetryRecorder, TelemetryStore,
                               estimate_task, pick_env, prior_table, format_expectation)

`agent_elapsed.telemetry` 里还有一个可选依赖：它把 hook 生命周期翻译成遥测行，
是"模型会越用越准"所需要的、但又不是纯数学生成的那一段。
"""
from __future__ import annotations

from agent_elapsed.constants import (
    ConstTable, SchemaMismatch, bundled_prior_tables, load_tables, pick_env, prior_table,
)
from agent_elapsed.estimate import EstResult, estimate_task
from agent_elapsed.hooks import DisplayPolicy, ElapsedHooks, TaskOutcome, TaskView, WidgetState
from agent_elapsed.presentation import format_elapsed, format_expectation
from agent_elapsed.schema import StepEvent, TaskRecord
from agent_elapsed.store import TelemetryStore
from agent_elapsed.telemetry import TelemetryRecorder

__all__ = [
    "ConstTable", "DisplayPolicy", "ElapsedHooks", "EstResult", "SchemaMismatch", "StepEvent",
    "TaskOutcome", "TaskRecord", "TaskView", "TelemetryRecorder", "TelemetryStore", "WidgetState",
    "bundled_prior_tables", "estimate_task", "format_elapsed", "format_expectation",
    "load_tables", "pick_env", "prior_table",
]
