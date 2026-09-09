"""The ``ErasableStore`` protocol and the result types every adapter returns.

An adapter never decides an outcome. It suppresses, reclaims, and answers probes with facts
(found / not found, counts, file names). The status lattice in ``verify/levels.py`` decides.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from tombstone.model.artifacts import ArtifactRef
from tombstone.model.status import VerifyLevel


@dataclass(frozen=True, slots=True)
class ProbeSet:
    """Probe queries for one artifact: vectors reconstructed from the fingerprint and/or the
    probe table. ``k`` is the top-k depth. Never contains text."""

    artifact_id: str
    vectors: tuple[tuple[float, ...], ...] = ()
    k: int = 40


@dataclass(frozen=True, slots=True)
class LogicalProbeResult:
    found: bool
    found_by: tuple[str, ...] = ()  # "id", "filter", "topk:<i>", "mmr:<i>", "raw:<i>"
    probes_run: int = 0
    detail: str = ""

    def __bool__(self) -> bool:
        return self.found


@dataclass(frozen=True, slots=True)
class PhysicalProbeResult:
    found: bool
    method: str  # "byte-scan", "pgstattuple+heap-scan", "hash-scan", ...
    locations: tuple[str, ...] = ()  # files / relations where a match was found
    measurement: Mapping[str, float] = field(default_factory=dict)
    detail: str = ""

    def __bool__(self) -> bool:
        return self.found


@dataclass(frozen=True, slots=True)
class ReclaimResult:
    noop: bool
    method: str
    measurement: Mapping[str, float] = field(default_factory=dict)
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Hit:
    key: str
    score: float
    metadata: Mapping[str, Any]
    document: str | None = None


@runtime_checkable
class ErasableStore(Protocol):
    name: str
    kind: str
    capabilities: frozenset[VerifyLevel]  # what THIS instance can verify. Managed pg → {LOGICAL}

    def suppress(self, refs: Sequence[ArtifactRef]) -> None: ...  # phase 1, idempotent
    def reclaim(self, refs: Sequence[ArtifactRef]) -> ReclaimResult: ...  # phase 2, idempotent
    def probe_logical(self, ref: ArtifactRef, probes: ProbeSet) -> LogicalProbeResult: ...
    def probe_physical(self, ref: ArtifactRef) -> PhysicalProbeResult: ...  # NotSupported if not in caps
    def count(self) -> int: ...
    def sample_keys(self, n: int) -> list[str]: ...
    def version(self) -> str: ...
    def close(self) -> None: ...


def capability_note(caps: frozenset[VerifyLevel]) -> str:
    return "{" + ", ".join(sorted(c.value for c in caps)) + "}"
