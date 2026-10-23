"""Long-lived UTF-8 event bridge for hosts written in any language."""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading

from agent_elapsed.constants import ConstTable, load_tables, pick_env
from agent_elapsed.hooks import DisplayPolicy, ElapsedHooks, TaskOutcome


def _task_id(event: dict) -> str:
    task_id = event.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a nonempty string")
    return task_id


def handle_event(hooks: ElapsedHooks, event: dict) -> TaskOutcome | None:
    if not isinstance(event, dict) or event.get("v", 1) != 1:
        raise ValueError("event must be an object using protocol v=1")
    name = event.get("event")
    if name == "message":
        return None
    if name == "prompt_received":
        if event.get("intent") != "task":
            return None
        name = "task_started"
    if name == "task_started":
        g = event.get("g")
        if g is not None and not isinstance(g, dict):
            raise ValueError("g must be an object or null")
        hooks.on_task_start(_task_id(event), event.get("title"), g=g,
                            probs=event.get("probs", ()), long_task=event.get("long_task"))
        return None
    endings = {"task_finished": hooks.on_task_finish,
               "task_cancelled": hooks.on_task_cancel,
               "task_failed": hooks.on_task_fail}
    if name in endings:
        return endings[name](_task_id(event))
    if name == "tick":
        hooks.refresh()
        return None
    if name == "shutdown":
        hooks.close()
        return None
    raise ValueError(f"unknown event: {name!r}")


def run_bridge(source, output, *, table: ConstTable,
               policy: DisplayPolicy | None = None, tick_seconds: float = 1.0):
    if not math.isfinite(tick_seconds) or tick_seconds <= 0:
        raise ValueError("tick_seconds must be finite and positive")
    stop = threading.Event()
    write_lock = threading.Lock()
    last_state = None

    def emit(payload):
        with write_lock:
            output.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()

    def update(state):
        nonlocal last_state
        data = state.to_dict()
        if data != last_state:
            emit(data)
            last_state = data

    with ElapsedHooks(table, policy=policy) as hooks:
        hooks.subscribe(update)

        def tick():
            while not stop.wait(tick_seconds):
                hooks.refresh()

        ticker = threading.Thread(target=tick, name="elapsed-bridge-tick", daemon=True)
        ticker.start()
        try:
            for line in source:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    outcome = handle_event(hooks, event)
                    if outcome:
                        emit({"v": 1, "event": "elapsed.finished", "task": outcome.to_dict()})
                    if event.get("event") == "shutdown":
                        break
                except (ValueError, TypeError, RecursionError) as exc:
                    emit({"v": 1, "event": "elapsed.error", "message": str(exc)})
        finally:
            stop.set()
            ticker.join()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constants", default="constants.json")
    parser.add_argument("--env", default="default")
    parser.add_argument("--long-task-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    # Pipes must use a stable encoding regardless of the host's Windows locale.
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        table = pick_env(load_tables(args.constants), args.env)
        run_bridge(sys.stdin, sys.stdout, table=table,
                   policy=DisplayPolicy(args.long_task_seconds))
    except (OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def console() -> None:
    """Entry point for the `agent-elapsed-bridge` script."""
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
