"""把 hook 生命周期落成遥测 —— 自校正闭环里原先缺的那一段。

`ElapsedHooks` 只负责"显示"：它知道任务何时开始、何时结束、用了多久、预估算没算出来。
`TelemetryStore` 只负责"落盘"：它只认已校验的行。两边都在，但中间没有任何东西把
`TaskOutcome` 翻译成任务记录，也没人记步事件 —— 于是"模型会越用越准"这句话在代码里
无法成立，只能靠宿主自己从头写适配器。

`TelemetryRecorder` 就是那个适配器：

    recorder = TelemetryRecorder(hooks, store, model_id="my-model", host="my-host")
    recorder.start("t-1", "修复接口并运行测试", g=g, probs=[("patch", .7)], long_task=True)
    with recorder.step("t-1", "llm"):
        ...
    with recorder.step("t-1", "tool", tool="pytest"):
        ...
    recorder.finish("t-1")        # cancel() 标记删失，fail() 记失败

设计取舍，都不是随手定的：

* **步事件立刻写盘**，不等任务结束：宿主崩溃或用户取消时，已经发生的步骤本身就是完整观测，
  正是删失处理需要的那批数据。
* **取消 = 删失（censored=True）**：取消任务的 total_wall 只是下界，不能拿去算计数或覆盖率，
  但它的步长仍然有效。
* **失败 = 完整观测**：任务确实结束并耗掉了那些时间。
* **没有分类也要落盘**：`g` 缺失时记录照写，`g_pred` 为空字典。离线侧会用
  `offline.partition.partition_tasks` 把它隔离并计数，而不是让整条拟合因为一条缺列记录崩掉。
* 同一个任务重复终结是幂等的；`store` 拒绝的行不会抛异常，只返回 False。
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator

from agent_elapsed.features import G_SCHEMA_VERSION
from agent_elapsed.hooks import ElapsedHooks, TaskOutcome
from agent_elapsed.store import TelemetryStore


@dataclass
class TelemetryRecorder:
    hooks: ElapsedHooks
    store: TelemetryStore
    model_id: str
    host: str
    clock: Callable[[], float] = time.monotonic
    _features: dict = field(default_factory=dict)      # task_id -> g
    _probs: dict = field(default_factory=dict)         # task_id -> ((level, weight), ...)
    _recorded: set = field(default_factory=set)

    def __post_init__(self):
        for name in ("model_id", "host"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} 必须是非空字符串：它构成环境键的一半")

    # ---- 生命周期 --------------------------------------------------------------
    def start(self, task_id: str, title: str, *, g: dict | None = None, probs=(),
              long_task: bool | None = None):
        """Start the widget timer and remember the features this task was predicted with."""
        self.hooks.on_task_start(task_id, title, g=g, probs=probs, long_task=long_task)
        self._features[task_id] = dict(g) if isinstance(g, dict) else None
        self._probs[task_id] = tuple((str(level), float(weight)) for level, weight in (probs or ()))

    def finish(self, task_id: str) -> TaskOutcome | None:
        return self._close(task_id, self.hooks.on_task_finish, censored=False)

    def cancel(self, task_id: str) -> TaskOutcome | None:
        return self._close(task_id, self.hooks.on_task_cancel, censored=True)

    def fail(self, task_id: str) -> TaskOutcome | None:
        return self._close(task_id, self.hooks.on_task_fail, censored=False)

    @contextmanager
    def step(self, task_id: str, op: str, *, ok: bool = True,
             tool: str | None = None) -> Iterator[dict]:
        """Time one step and write it immediately.

        The yielded dict is mutable: set `state["ok"] = False` to record a failed step without
        losing its duration (a failed step still happened and still cost that time).
        """
        started = self.clock()
        state = {"ok": bool(ok)}
        try:
            yield state
        finally:
            self.store.append_step({
                "v": G_SCHEMA_VERSION, "task_id": task_id, "op": op,
                "t_start": float(started), "t_end": float(self.clock()),
                "ok": bool(state.get("ok", True)), "tool": tool,
            })

    # ---- 落盘 ------------------------------------------------------------------
    def _close(self, task_id: str, ender, *, censored: bool) -> TaskOutcome | None:
        outcome = ender(task_id)
        if outcome is None:
            return None
        return self.record(outcome, censored=censored)

    def record(self, outcome: TaskOutcome, *, censored: bool | None = None) -> TaskOutcome:
        """Write one task record; idempotent per task id."""
        if outcome.task_id in self._recorded:
            return outcome
        is_censored = bool(outcome.status == "cancelled") if censored is None else bool(censored)
        features = self._features.pop(outcome.task_id, None)
        self.store.append_task({
            "v": G_SCHEMA_VERSION,
            "task_id": outcome.task_id,
            "g_pred": features or {},
            "g_pred_probs": [list(pair) for pair in self._probs.pop(outcome.task_id, ())],
            "g_final": None,
            "total_wall": float(outcome.total_wall),
            "censored": is_censored,
            "model_id": self.model_id,
            "host": self.host,
            "estimator_wall": float(outcome.estimator_wall),
            "g_schema_version": G_SCHEMA_VERSION,
        })
        self._recorded.add(outcome.task_id)
        return outcome

    def close(self) -> None:
        self.hooks.close()
