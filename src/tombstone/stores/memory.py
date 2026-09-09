"""An in-memory dict store. The fifth adapter written from ``docs/adapters.md`` alone (Phase 8.5)
and the store used by LangGraph memory entries (Phase 8.2). Nothing is persisted, so PHYSICAL is
trivially the same as LOGICAL: the bytes live only in this process's dict."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tombstone.model.artifacts import ArtifactRef
from tombstone.model.status import VerifyLevel
from tombstone.stores.base import (
    LogicalProbeResult,
    PhysicalProbeResult,
    ProbeSet,
    ReclaimResult,
)


class MemoryStore:
    kind = "memory"

    def __init__(self, name: str) -> None:
        self.name = name
        self._rows: dict[str, dict[str, Any]] = {}
        self._suppressed: set[str] = set()
        self.capabilities: frozenset[VerifyLevel] = frozenset(
            {VerifyLevel.LOGICAL, VerifyLevel.PHYSICAL}
        )

    def version(self) -> str:
        return "memory 1"

    def close(self) -> None:
        return None

    def put(self, key: str, row: dict[str, Any]) -> None:
        self._rows[key] = dict(row)
        self._suppressed.discard(key)

    def get(self, key: str) -> dict[str, Any] | None:
        if key in self._suppressed:
            return None
        return self._rows.get(key)

    def suppress(self, refs: Sequence[ArtifactRef]) -> None:
        self._suppressed.update(r.store_key for r in refs)

    def reclaim(self, refs: Sequence[ArtifactRef]) -> ReclaimResult:
        n = 0
        for r in refs:
            if self._rows.pop(r.store_key, None) is not None:
                n += 1
        return ReclaimResult(noop=n == 0, method="dict.pop", measurement={"deleted": float(n)})

    def probe_logical(self, ref: ArtifactRef, probes: ProbeSet) -> LogicalProbeResult:
        found = self.get(ref.store_key) is not None
        return LogicalProbeResult(found=found, found_by=("id",) if found else (), probes_run=1)

    def probe_physical(self, ref: ArtifactRef) -> PhysicalProbeResult:
        found = ref.store_key in self._rows
        return PhysicalProbeResult(
            found=found, method="dict membership", locations=("memory",) if found else ()
        )

    def count(self) -> int:
        return len(self._rows)

    def sample_keys(self, n: int) -> list[str]:
        return sorted(self._rows)[:n]
