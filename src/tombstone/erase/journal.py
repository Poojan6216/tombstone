"""The saga journal: every step recorded *before* it runs, hash-chained, resumable.

Record types:
  saga_start   {saga_id, trace_id, subject, scope, reason, artifact_count}
  step_begin   {saga_id, step_id, phase, store, artifact_ids}
  step_end     {saga_id, step_id, ok, noop, error?, result}
  probe        {saga_id, artifact_id, level, probe, found, measurement}
  dlq          {saga_id, step_id, store, artifact_ids, error, retry}
  saga_end     {saga_id, receipt_id}

Bodies carry ids, names, counts and measurements — never content (Hard Rule 7).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tombstone.chain import HashChain, Record


class Journal:
    SAGA_START = "saga_start"
    STEP_BEGIN = "step_begin"
    STEP_END = "step_end"
    PROBE = "probe"
    DLQ = "dlq"
    SAGA_END = "saga_end"

    def __init__(self, path: Path, lock_timeout_s: float = 30.0) -> None:
        self.chain = HashChain(path, lock_timeout_s)
        self.path = Path(path)

    def append(self, type_: str, body: dict[str, Any]) -> Record:
        return self.chain.append(type_, body)

    def records(self, saga_id: str | None = None) -> list[Record]:
        recs = self.chain.read()
        if saga_id is None:
            return recs
        return [r for r in recs if r.body.get("saga_id") == saga_id]

    def head(self) -> str:
        return self.chain.head()

    def verify(self) -> int:
        return self.chain.verify()

    # --- convenience writers ------------------------------------------------------------------

    def saga_start(
        self,
        saga_id: str,
        trace_id: str,
        subject_hmac: str,
        scope: str,
        reason: str,
        artifact_count: int,
        extra: dict[str, Any] | None = None,
    ) -> Record:
        body: dict[str, Any] = {
            "saga_id": saga_id,
            "trace_id": trace_id,
            "subject": subject_hmac,
            "scope": scope,
            "reason": reason,
            "artifact_count": artifact_count,
        }
        if extra:
            body.update(extra)
        return self.append(self.SAGA_START, body)

    def step_begin(
        self, saga_id: str, step_id: str, phase: str, store: str, artifact_ids: Sequence[str]
    ) -> Record:
        return self.append(
            self.STEP_BEGIN,
            {
                "saga_id": saga_id,
                "step_id": step_id,
                "phase": phase,
                "store": store,
                "artifact_ids": list(artifact_ids),
            },
        )

    def step_end(
        self,
        saga_id: str,
        step_id: str,
        ok: bool,
        noop: bool = False,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> Record:
        body: dict[str, Any] = {
            "saga_id": saga_id,
            "step_id": step_id,
            "ok": ok,
            "noop": noop,
            "result": result or {},
        }
        if error:
            body["error"] = error
        return self.append(self.STEP_END, body)

    def probe(
        self,
        saga_id: str,
        artifact_id: str,
        level: str,
        probe: str,
        found: bool | None,
        measurement: dict[str, float] | None = None,
        detail: str = "",
    ) -> Record:
        return self.append(
            self.PROBE,
            {
                "saga_id": saga_id,
                "artifact_id": artifact_id,
                "level": level,
                "probe": probe,
                "found": found,
                "measurement": {k: float(v) for k, v in sorted((measurement or {}).items())},
                "detail": detail,
            },
        )

    def dlq(
        self,
        saga_id: str,
        step_id: str,
        store: str,
        artifact_ids: Sequence[str],
        error: str,
        retry: int,
    ) -> Record:
        return self.append(
            self.DLQ,
            {
                "saga_id": saga_id,
                "step_id": step_id,
                "store": store,
                "artifact_ids": list(artifact_ids),
                "error": error,
                "retry": retry,
            },
        )

    def saga_end(self, saga_id: str, receipt_id: str) -> Record:
        return self.append(self.SAGA_END, {"saga_id": saga_id, "receipt_id": receipt_id})

    # --- resume support --------------------------------------------------------------------------

    def completed_steps(self, saga_id: str) -> dict[str, Record]:
        return {
            r.body["step_id"]: r
            for r in self.records(saga_id)
            if r.type == self.STEP_END and r.body.get("ok")
        }

    def begun_steps(self, saga_id: str) -> dict[str, Record]:
        return {r.body["step_id"]: r for r in self.records(saga_id) if r.type == self.STEP_BEGIN}

    def open_sagas(self) -> list[str]:
        started = [r.body["saga_id"] for r in self.records() if r.type == self.SAGA_START]
        ended = {r.body["saga_id"] for r in self.records() if r.type == self.SAGA_END}
        return [s for s in started if s not in ended]

    def saga_for_trace(self, trace_id: str) -> str | None:
        for r in self.records():
            if r.type == self.SAGA_START and r.body.get("trace_id") == trace_id:
                return str(r.body["saga_id"])
        return None
