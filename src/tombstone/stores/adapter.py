"""A LoRA adapter as an ``ErasableStore``. Implemented in Phase 4; this module exists so the
registry can name it. Everything delegates to ``tombstone.train``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tombstone.errors import NotSupported
from tombstone.model.artifacts import ArtifactRef
from tombstone.model.status import VerifyLevel
from tombstone.stores.base import (
    LogicalProbeResult,
    PhysicalProbeResult,
    ProbeSet,
    ReclaimResult,
)


class AdapterStore:
    kind = "adapter"

    def __init__(
        self,
        name: str,
        path: str | Path,
        shards: int = 1,
        base_model: str = "",
        runtime_model_cfg: Any = None,
    ) -> None:
        self.name = name
        self.path = Path(path)
        self.shards = shards
        self.base_model = base_model
        self.model_cfg = runtime_model_cfg
        self.capabilities: frozenset[VerifyLevel] = frozenset({VerifyLevel.LOGICAL})

    def version(self) -> str:
        return "adapter (phase 4 pending)"

    def close(self) -> None:
        return None

    def _pending(self) -> NotSupported:
        return NotSupported(
            f"adapter store {self.name!r}: the model leg is implemented in Phase 4 (train extra)"
        )

    def suppress(self, refs: Sequence[ArtifactRef]) -> None:
        raise self._pending()

    def reclaim(self, refs: Sequence[ArtifactRef]) -> ReclaimResult:
        raise self._pending()

    def probe_logical(self, ref: ArtifactRef, probes: ProbeSet) -> LogicalProbeResult:
        raise self._pending()

    def probe_physical(self, ref: ArtifactRef) -> PhysicalProbeResult:
        raise self._pending()

    def count(self) -> int:
        return 0

    def sample_keys(self, n: int) -> list[str]:
        return []
