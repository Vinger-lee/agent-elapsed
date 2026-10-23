"""JSONL 追加写。运行时只写不读；读接口是给离线侧和测试用的。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agent_elapsed.schema import (
    StepEvent, TaskRecord, parse_step_row, parse_task_row,
)


@dataclass
class TelemetryStore:
    events_path: Path
    tasks_path: Path
    rejects_path: Path
    _seen: set = field(default_factory=set)

    def __post_init__(self):
        self.events_path = Path(self.events_path)
        self.tasks_path = Path(self.tasks_path)
        self.rejects_path = Path(self.rejects_path)

    def _append(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def append_step(self, row: dict) -> bool:
        event, reason = parse_step_row(row, self._seen)
        if event is None:
            self._append(self.rejects_path, {"reason": reason, "kind": "step", "row": row})
            return False
        self._append(self.events_path, asdict(event))
        return True

    def append_task(self, row: dict) -> bool:
        record, reason = parse_task_row(row)
        if record is None:
            self._append(self.rejects_path, {"reason": reason, "kind": "task", "row": row})
            return False
        payload = asdict(record)
        payload["g_pred_probs"] = [list(p) for p in record.g_pred_probs]
        self._append(self.tasks_path, payload)
        return True

    def read_steps(self) -> list[StepEvent]:
        seen, out = set(), []
        for row in self._read_rows(self.events_path, 'step'):
            try:
                event, reason = parse_step_row(row, seen)
            except (KeyError, TypeError, ValueError, OverflowError):
                event, reason = None, 'malformed_row'
            if event is None:
                self._append(self.rejects_path, {'kind':'step','reason':reason,'row':row})
            else:
                out.append(event)
        self._seen.update(seen)
        return out

    def read_tasks(self) -> list[TaskRecord]:
        out = []
        for row in self._read_rows(self.tasks_path, 'task'):
            try:
                record, reason = parse_task_row(row)
            except (KeyError, TypeError, ValueError, OverflowError):
                record, reason = None, 'malformed_row'
            if record is None:
                self._append(self.rejects_path, {'kind':'task','reason':reason,'row':row})
            else:
                out.append(record)
        return out

    def _read_rows(self, path: Path, kind: str):
        for line in self._lines(path):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                self._append(self.rejects_path, {'kind':kind,'reason':'invalid_json','row':line})
                continue
            if not isinstance(row, dict):
                self._append(self.rejects_path, {'kind':kind,'reason':'malformed_row','row':row})
                continue
            yield row

    @staticmethod
    def _lines(path: Path) -> list[str]:
        if not path.exists():
            return []
        return [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
