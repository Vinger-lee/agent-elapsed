"""Host task lifecycle and embedded-widget state; no host UI or I/O ownership."""
from __future__ import annotations

import logging
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from agent_elapsed.constants import ConstTable
from agent_elapsed.estimate import EstResult, estimate_task
from agent_elapsed.presentation import format_elapsed, format_expectation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DisplayPolicy:
    long_task_seconds: float = 60.0

    def __post_init__(self):
        value = self.long_task_seconds
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("long_task_seconds must be a finite positive number")


@dataclass(frozen=True)
class TaskView:
    task_id: str
    title: str
    elapsed_seconds: float
    estimate: EstResult | None

    def to_dict(self) -> dict:
        calibrated = self.estimate is not None and not self.estimate.calibrating
        return {
            "task_id": self.task_id, "title": self.title, "status": "running",
            "elapsed_seconds": self.elapsed_seconds,
            "elapsed_label": format_elapsed(self.elapsed_seconds),
            # 键名保持 band（宿主协议不轻改），内容是"中位数 + 九成上界"，
            # 不是把 40% 概率的 P50–P90 区间说成"大约落在这里"。
            "band": (format_expectation(self.estimate.p50, self.estimate.p90)
                     if calibrated else None),
            "calibrating": self.estimate.calibrating if self.estimate else None,
            "over_estimate": bool(calibrated and self.elapsed_seconds > self.estimate.p90),
            "estimate": self.estimate.to_dict() if self.estimate else None,
        }


@dataclass(frozen=True)
class WidgetState:
    tasks: tuple[TaskView, ...] = ()

    def to_dict(self) -> dict:
        return {"v": 1, "event": "elapsed.updated",
                "tasks": [task.to_dict() for task in self.tasks]}


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    status: str
    total_wall: float
    estimate: EstResult | None
    estimator_wall: float
    estimation_error: str | None

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "status": self.status,
                "total_wall": self.total_wall, "estimator_wall": self.estimator_wall,
                "estimate": self.estimate.to_dict() if self.estimate else None,
                "estimation_error": self.estimation_error}


@dataclass
class _Task:
    task_id: str
    title: str
    started_at: float
    long_task: bool | None
    estimate: EstResult | None = None
    estimator_wall: float = 0.0
    estimation_error: str | None = None
    future: Future | None = None


class ElapsedHooks:
    """One instance per host session. Only explicit task events start timers.

    Call refresh() from the host's timer. Subscribers may run on an estimation
    thread; dispatch their updates onto the host's UI thread as needed.
    """

    def __init__(self, table: ConstTable, *, policy: DisplayPolicy | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 estimator: Callable[..., EstResult] = estimate_task):
        self.table = table
        self.policy = policy or DisplayPolicy()
        self._clock = clock
        self._estimator = estimator
        self._tasks: dict[str, _Task] = {}
        self._seen: set[str] = set()
        self._subscribers: list[Callable[[WidgetState], None]] = []
        self._lock = threading.RLock()
        self._publish_lock = threading.RLock()
        self._publish_revision = 0
        self._delivery = threading.local()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-elapsed")
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def on_task_start(self, task_id: str, title: str, *, g: dict | None = None,
                      probs=(), long_task: bool | None = None) -> Future | None:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a nonempty string")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a nonempty string")
        if long_task is not None and not isinstance(long_task, bool):
            raise ValueError("long_task must be true, false, or null")
        # Copy before mutating the registry: invalid input must not leave a task.
        features = dict(g) if g is not None else None
        probabilities = tuple(tuple(pair) for pair in (probs or ()))
        with self._lock:
            if self._closed:
                raise RuntimeError("ElapsedHooks is closed")
            if task_id in self._tasks:
                return self._tasks[task_id].future
            if task_id in self._seen:
                raise ValueError("task_id has already finished; use a new unique ID")
            task = _Task(task_id, title, self._clock(), long_task)
            self._tasks[task_id] = task
            self._seen.add(task_id)
            if features is not None:
                task.future = self._pool.submit(self._estimate, task, features, probabilities)
        self._publish()
        return task.future

    def _estimate(self, task: _Task, g: dict, probs) -> EstResult | None:
        started = time.perf_counter()
        result, error = None, None
        try:
            result = self._estimator(g, probs, table=self.table)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        duration = time.perf_counter() - started
        with self._lock:
            if self._closed or self._tasks.get(task.task_id) is not task:
                return result
            task.estimate, task.estimation_error = result, error
            task.estimator_wall = duration
        self._publish()
        return result

    def snapshot(self) -> WidgetState:
        with self._lock:
            now = self._clock()
            visible = []
            for task in self._tasks.values():
                elapsed = max(0.0, now - task.started_at)
                predicted_long = (task.long_task is not False and task.estimate is not None
                                  and not task.estimate.calibrating
                                  and task.estimate.p50 >= self.policy.long_task_seconds)
                if task.long_task is True or predicted_long or elapsed >= self.policy.long_task_seconds:
                    visible.append(TaskView(task.task_id, task.title, elapsed, task.estimate))
            return WidgetState(tuple(visible))

    def refresh(self) -> WidgetState:
        return self._publish()

    def subscribe(self, callback: Callable[[WidgetState], None]) -> Callable[[], None]:
        with self._publish_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("ElapsedHooks is closed")
                self._subscribers.append(callback)
            self._deliver(callback, self.snapshot())

        def unsubscribe():
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def _deliver(self, callback, state):
        depth = getattr(self._delivery, "depth", 0)
        self._delivery.depth = depth + 1
        try:
            callback(state)
        except Exception:
            logger.exception("elapsed widget subscriber failed")
        finally:
            self._delivery.depth = depth

    def _publish(self) -> WidgetState:
        # Take the snapshot inside the publication lock so an older worker
        # update cannot be delivered after a newer completion update.
        with self._publish_lock:
            self._publish_revision += 1
            revision = self._publish_revision
            state = self.snapshot()
            with self._lock:
                subscribers = tuple(self._subscribers)
            for callback in subscribers:
                # A subscriber may end a task and publish again synchronously.
                # Its newer state supersedes the rest of this delivery.
                if revision != self._publish_revision:
                    break
                self._deliver(callback, state)
            return state if revision == self._publish_revision else self.snapshot()

    def _finish(self, task_id: str, status: str) -> TaskOutcome | None:
        with self._lock:
            task = self._tasks.pop(task_id, None)
            if task is None:
                return None
            outcome = TaskOutcome(task_id, status, max(0.0, self._clock() - task.started_at),
                                  task.estimate, task.estimator_wall, task.estimation_error)
            if task.future:
                task.future.cancel()
        self._publish()
        return outcome

    def on_task_finish(self, task_id: str) -> TaskOutcome | None:
        return self._finish(task_id, "completed")

    def on_task_cancel(self, task_id: str) -> TaskOutcome | None:
        return self._finish(task_id, "cancelled")

    def on_task_fail(self, task_id: str) -> TaskOutcome | None:
        return self._finish(task_id, "failed")

    def close(self, *, wait: bool = True):
        """Clear state and cancel queued work; close inside a callback never joins."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._tasks.clear()
        self._publish()
        with self._lock:
            self._subscribers.clear()
        # A worker may need the publication lock held by this callback, or may
        # be the callback's own thread. Let it unwind rather than joining it.
        self._pool.shutdown(wait=wait and not getattr(self._delivery, "depth", 0),
                            cancel_futures=True)
