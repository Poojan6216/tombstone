"""An in-repo semantic cache on any supported vector backend.

``lookup(query)`` embeds the query and returns the cached answer of the nearest entry when the
cosine similarity clears ``threshold``. ``update(query, answer, chunk_ids)`` stores the answer as
a record whose vector is the query embedding, and creates a CACHE node with one edge per parent
chunk. The record's fingerprint is the query embedding, so the physical probe is a byte scan.

Suppression deletes the entry (and, optionally, its ``purge_k`` nearest neighbours — the
mitigation for paraphrased hits, measured in Phase 7.3). Reclaim delegates to the backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tombstone.embeddings import Embedder
from tombstone.lineage.capture import Capture, EmbedRecord
from tombstone.lineage.stamp import K_CHUNK, K_EMBED
from tombstone.model.artifacts import ArtifactRef
from tombstone.model.lineage import Node
from tombstone.model.status import VerifyLevel
from tombstone.stores._vector import VectorBackendBase
from tombstone.stores.base import (
    LogicalProbeResult,
    PhysicalProbeResult,
    ProbeSet,
    ReclaimResult,
)
from tombstone.util import fingerprint_f32, sha256_hex

K_ANSWER_HASH = "tombstone.answer_hash"


class SemanticCache:
    kind = "cache_semantic"

    def __init__(
        self,
        name: str,
        backing: VectorBackendBase,
        embedder: Embedder,
        capture: Capture,
        threshold: float = 0.9,
        purge_k: int = 0,
    ) -> None:
        self.name = name
        self.backing = backing
        self.embedder = embedder
        self.capture = capture
        self.threshold = threshold
        self.purge_k = purge_k
        capture.register(name, self.kind)
        self.capabilities: frozenset[VerifyLevel] = frozenset(
            backing.capabilities & {VerifyLevel.LOGICAL, VerifyLevel.PHYSICAL}
        )

    def version(self) -> str:
        return f"semantic-cache on {self.backing.version()}"

    def close(self) -> None:
        self.backing.close()

    # --- app-facing ------------------------------------------------------------------------------

    def lookup(self, query: str) -> tuple[str, dict[str, Any]] | None:
        vec = self.embedder.embed([query])[0]
        hits = self.backing.query(vec, 1)
        if not hits:
            return None
        h = hits[0]
        sim = _similarity(h.score, self.backing.kind)
        if sim < self.threshold:
            return None
        return (h.document or ""), dict(h.metadata)

    def update(
        self, query: str, answer: str, chunk_ids: Sequence[str], subject_hmac: str | None = None
    ) -> list[Node]:
        vec = self.embedder.embed([query])[0]
        key = sha256_hex(query)[:32]
        answer_hash = sha256_hex(answer)
        nodes = self.capture.record_cache(
            self.name,
            key,
            list(chunk_ids),
            answer_hash,
            subject=None,
            fingerprint=fingerprint_f32(vec),
            via="cache:semantic",
        )
        md: dict[str, Any] = {
            K_ANSWER_HASH: answer_hash,
            K_CHUNK: ",".join(chunk_ids),
            K_EMBED: ",".join(n.artifact_id for n in nodes),
            "tombstone.scope": self.capture.scope.tenant,
        }
        rec = EmbedRecord(
            key=key,
            vector=[float(x) for x in vec],
            metadata=md,
            document=answer,
            embed_node=nodes[0] if nodes else _placeholder(),
            chunk_node=nodes[0] if nodes else _placeholder(),
        )
        self.backing.add([rec])
        return nodes

    # --- ErasableStore -------------------------------------------------------------------------

    @staticmethod
    def _key_of(ref: ArtifactRef) -> str:
        return ref.store_key.split("@", 1)[0]

    def _neighbour_keys(self, key: str) -> list[str]:
        if self.purge_k <= 0:
            return []
        vec = self.backing._vector_of(key)
        if vec is None:
            return []
        return [h.key for h in self.backing.query(vec, self.purge_k + 1) if h.key != key][
            : self.purge_k
        ]

    def suppress(self, refs: Sequence[ArtifactRef]) -> None:
        keys = [self._key_of(r) for r in refs]
        extra: list[str] = []
        for k in keys:
            extra.extend(self._neighbour_keys(k))
        self.backing.native_delete(list(dict.fromkeys(keys + extra)))

    def reclaim(self, refs: Sequence[ArtifactRef]) -> ReclaimResult:
        fake = [
            ArtifactRef(
                artifact_id=r.artifact_id,
                kind=r.kind,
                store=self.backing.name,
                store_key=self._key_of(r),
                scope=r.scope,
                content_hash=r.content_hash,
                embedding_fingerprint=r.embedding_fingerprint,
            )
            for r in refs
        ]
        res = self.backing.reclaim(fake)
        return ReclaimResult(
            noop=res.noop,
            method=f"invalidate + purge {self.purge_k} neighbours",
            measurement=dict(res.measurement),
        )

    def probe_logical(self, ref: ArtifactRef, probes: ProbeSet) -> LogicalProbeResult:
        key = self._key_of(ref)
        found_by: list[str] = []
        if key in self.backing.get([key]):
            found_by.append("id")
        for i, v in enumerate(probes.vectors):
            hits = self.backing.query(v, 1)
            if (
                hits
                and hits[0].key == key
                and _similarity(hits[0].score, self.backing.kind) >= self.threshold
            ):
                found_by.append(f"topk:{i}")
        return LogicalProbeResult(
            found=bool(found_by), found_by=tuple(found_by), probes_run=1 + len(probes.vectors)
        )

    def probe_physical(self, ref: ArtifactRef) -> PhysicalProbeResult:
        return self.backing.probe_physical(ref)

    def count(self) -> int:
        return self.backing.count()

    def sample_keys(self, n: int) -> list[str]:
        return self.backing.sample_keys(n)


def _similarity(score: float, backend_kind: str) -> float:
    """Normalise a backend score to cosine similarity."""
    if backend_kind in {"chroma", "pgvector"}:  # cosine distance
        return 1.0 - score
    return score  # faiss inner product on unit vectors, qdrant cosine score


def _placeholder() -> Node:
    from tombstone.model.artifacts import ArtifactKind, Scope

    return Node("", ArtifactKind.CACHE, "", "", Scope("default"), "", None, "", 0)
