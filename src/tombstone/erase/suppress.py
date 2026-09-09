"""Phase 1: suppression. Write tombstone rows in lineage (honoured by every wrapper query path
immediately), then ask each store for its fastest available suppression. Idempotent."""

from __future__ import annotations

from collections.abc import Sequence

from tombstone.lineage.store import LineageStore
from tombstone.model.artifacts import ArtifactRef
from tombstone.stores.base import ErasableStore


def tombstone_in_lineage(
    lineage: LineageStore, refs: Sequence[ArtifactRef], trace_id: str, reason: str
) -> int:
    """The durable marker. Returns the lineage seq it was written at."""
    return lineage.tombstone(
        [r.artifact_id for r in refs], reason=f"erase:{reason}", trace_id=trace_id
    )


def suppress_in_store(store: ErasableStore, refs: Sequence[ArtifactRef]) -> int:
    store.suppress(list(refs))
    return len(refs)
