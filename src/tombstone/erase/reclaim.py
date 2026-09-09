"""Phase 2: physical reclaim, delegated to each store's ``reclaim``. Idempotent per store.

pgvector: DELETE + REINDEX INDEX + VACUUM FULL (or VACUUM + REINDEX when FULL is denied — the
method string records which). Credit: this is the recipe ``vector-forget`` uses for pgvector.
Chroma: delete, then rewrite the collection's segment from survivors, purge the queue, VACUUM.
FAISS: rebuild IndexHNSWFlat from survivors, atomic file replace. Qdrant: delete + rewrite
(local) or optimizer (server). Dataset: drop row, rewrite shard, re-hash manifest. Adapter:
Phase 4 (shard retrain / approximate unlearning).
"""

from __future__ import annotations

from collections.abc import Sequence

from tombstone.model.artifacts import ArtifactRef
from tombstone.stores.base import ErasableStore, ReclaimResult


def reclaim_in_store(store: ErasableStore, refs: Sequence[ArtifactRef]) -> ReclaimResult:
    return store.reclaim(list(refs))
