"""Shared machinery for the four vector backends.

Every query path (id lookup, metadata filter, top-k by vector, MMR, raw) goes through
``_query_raw`` + ``_filter_suppressed`` so suppression is honoured everywhere (Hard Rule 5).
Backends implement the storage primitives; this class implements the protocol on top of them.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tombstone.errors import NotSupported
from tombstone.lineage.capture import EmbedRecord
from tombstone.lineage.stamp import K_EMBED, K_SUPPRESSED
from tombstone.model.artifacts import ArtifactRef
from tombstone.model.status import VerifyLevel
from tombstone.stores.base import (
    Hit,
    LogicalProbeResult,
    PhysicalProbeResult,
    ProbeSet,
    ReclaimResult,
)
from tombstone.util import fingerprint_bytes
from tombstone.verify.physical import fingerprint_patterns, scan_files_for_patterns

OVERFETCH = 20


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=False))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


def mmr_select(
    query: Sequence[float],
    candidates: Sequence[tuple[str, Sequence[float]]],
    k: int,
    lambda_mult: float = 0.5,
) -> list[str]:
    """Maximal marginal relevance over candidate (key, vector) pairs. Pure."""
    if not candidates or k <= 0:
        return []
    sims = [cosine(query, v) for _, v in candidates]
    chosen: list[int] = [max(range(len(candidates)), key=lambda i: sims[i])]
    while len(chosen) < min(k, len(candidates)):
        best_i, best_score = -1, -math.inf
        for i in range(len(candidates)):
            if i in chosen:
                continue
            redundancy = max(cosine(candidates[i][1], candidates[j][1]) for j in chosen)
            score = lambda_mult * sims[i] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_i, best_score = i, score
        chosen.append(best_i)
    return [candidates[i][0] for i in chosen]


class VectorBackendBase(ABC):
    """Storage primitives are abstract; protocol methods are implemented here."""

    kind: str = "vector"

    def __init__(self, name: str, embedding_model: str, dims: int) -> None:
        self.name = name
        self.embedding_model = embedding_model
        self.dims = dims
        self._suppressed: set[str] = set()  # keys suppressed in this process (belt and braces)
        self.capabilities: frozenset[VerifyLevel] = frozenset({VerifyLevel.LOGICAL})

    # --- primitives ------------------------------------------------------------------------------

    @abstractmethod
    def _add(self, records: Sequence[EmbedRecord]) -> None: ...

    @abstractmethod
    def _get(self, keys: Sequence[str]) -> dict[str, Hit]:
        """Fetch by key regardless of suppression; metadata must include the stored flags."""

    @abstractmethod
    def _query_raw(self, vector: Sequence[float], k: int) -> list[Hit]:
        """Top-k by vector, without any suppression filtering."""

    @abstractmethod
    def _filter_raw(self, key: str, value: Any, k: int) -> list[Hit]:
        """Metadata equality filter, without suppression filtering."""

    @abstractmethod
    def _mark_suppressed(self, keys: Sequence[str]) -> None: ...

    @abstractmethod
    def _native_delete(self, keys: Sequence[str]) -> str:
        """The store's own delete. Returns a short description of what it did."""

    @abstractmethod
    def _reclaim(self, keys: Sequence[str]) -> ReclaimResult: ...

    @abstractmethod
    def _vector_of(self, key: str) -> list[float] | None: ...

    @abstractmethod
    def persisted_files(self) -> list[Path]: ...

    @abstractmethod
    def all_keys(self) -> list[str]: ...

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def close(self) -> None: ...

    def count(self) -> int:
        return len(self.all_keys())

    def sample_keys(self, n: int) -> list[str]:
        return sorted(self.all_keys())[:n]

    # --- app-facing ------------------------------------------------------------------------------

    def add(self, records: Sequence[EmbedRecord]) -> None:
        for r in records:
            r.metadata.setdefault(K_SUPPRESSED, False)
        self._add(records)

    def native_delete(self, keys: Sequence[str]) -> str:
        return self._native_delete(list(keys))

    def is_suppressed(self, hit: Hit) -> bool:
        return hit.key in self._suppressed or bool(hit.metadata.get(K_SUPPRESSED, False))

    def get(self, keys: Sequence[str], include_suppressed: bool = False) -> dict[str, Hit]:
        hits = self._get(keys)
        if include_suppressed:
            return hits
        return {k: h for k, h in hits.items() if not self.is_suppressed(h)}

    def query(self, vector: Sequence[float], k: int, include_suppressed: bool = False) -> list[Hit]:
        raw = self._query_raw(vector, k + OVERFETCH if not include_suppressed else k)
        if include_suppressed:
            return raw[:k]
        return [h for h in raw if not self.is_suppressed(h)][:k]

    def filter(self, key: str, value: Any, k: int = 100) -> list[Hit]:
        return [h for h in self._filter_raw(key, value, k) if not self.is_suppressed(h)][:k]

    def query_mmr(
        self, vector: Sequence[float], k: int, fetch_k: int = 20, lambda_mult: float = 0.5
    ) -> list[Hit]:
        cands = self.query(vector, max(fetch_k, k))
        vecs: list[tuple[str, Sequence[float]]] = []
        by_key: dict[str, Hit] = {}
        for h in cands:
            v = self._vector_of(h.key)
            if v is not None:
                vecs.append((h.key, v))
                by_key[h.key] = h
        order = mmr_select(vector, vecs, k, lambda_mult)
        return [by_key[key] for key in order]

    # --- ErasableStore ---------------------------------------------------------------------------

    def suppress(self, refs: Sequence[ArtifactRef]) -> None:
        keys = [r.store_key for r in refs]
        self._suppressed.update(keys)
        present = self._get(keys)
        self._mark_suppressed([k for k in keys if k in present])

    def reclaim(self, refs: Sequence[ArtifactRef]) -> ReclaimResult:
        keys = [r.store_key for r in refs]
        self._suppressed.update(keys)
        return self._reclaim(keys)

    def probe_logical(self, ref: ArtifactRef, probes: ProbeSet) -> LogicalProbeResult:
        found_by: list[str] = []
        run = 0
        # 1. id lookup
        run += 1
        if ref.store_key in self.get([ref.store_key]):
            found_by.append("id")
        # 2. metadata filter on the embed artifact id
        run += 1
        if any(h.key == ref.store_key for h in self.filter(K_EMBED, ref.artifact_id)):
            found_by.append("filter")
        # 3. top-k with each probe vector, then MMR
        for i, vec in enumerate(probes.vectors):
            run += 1
            if any(h.key == ref.store_key for h in self.query(vec, probes.k)):
                found_by.append(f"topk:{i}")
            run += 1
            if any(h.key == ref.store_key for h in self.query_mmr(vec, probes.k, probes.k * 2)):
                found_by.append(f"mmr:{i}")
        return LogicalProbeResult(found=bool(found_by), found_by=tuple(found_by), probes_run=run)

    def _physical_patterns(self, ref: ArtifactRef) -> dict[str, bytes]:
        patterns: dict[str, bytes] = {"artifact_id": ref.artifact_id.encode("utf-8")}
        if ref.embedding_fingerprint:
            patterns.update(self._encoding_patterns(ref.embedding_fingerprint))
        return patterns

    def _encoding_patterns(self, fingerprint_hex: str) -> dict[str, bytes]:
        """Which encodings this backend persists. Default: float32 little-endian only."""
        return {"f32le": fingerprint_bytes(fingerprint_hex)}

    def probe_physical(self, ref: ArtifactRef) -> PhysicalProbeResult:
        if VerifyLevel.PHYSICAL not in self.capabilities:
            raise NotSupported(
                f"store {self.name!r} cannot be physically verified: "
                f"{self.physical_unsupported_reason()}"
            )
        return scan_files_for_patterns(
            self.persisted_files(), self._physical_patterns(ref), method="byte-scan"
        )

    def physical_unsupported_reason(self) -> str:
        return "no filesystem access to the persisted index"

    # --- helpers for probes ----------------------------------------------------------------------

    def probe_vectors_for(self, ref: ArtifactRef) -> tuple[tuple[float, ...], ...]:
        """A probe vector reconstructed from the artifact's own stored vector when we still hold
        it (pre-deletion power check), else nothing — the caller adds probe-table vectors."""
        v = self._vector_of(ref.store_key)
        return (tuple(v),) if v is not None else ()

    @staticmethod
    def all_patterns(fingerprint_hex: str) -> Mapping[str, bytes]:
        return fingerprint_patterns(fingerprint_hex)
