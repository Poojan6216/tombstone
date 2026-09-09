"""Logical probes: is the artifact still retrievable by id, by filter, by top-k, by MMR?

Probe vectors (never text):
  * the padded fingerprint — the first 32 dims of the stored vector with zeros elsewhere (weak
    but free; ranks the target above unrelated neighbours in its own index);
  * the probe table — embeddings of three short queries derived from the chunk at capture time
    (first / middle / last words), stored as (query hash, embedding). The queries are fragments,
    so the table is a bounded leak; probe rows are purged once the artifact is VERIFIED.

The probe design is documented in ``docs/verification-levels.md``.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence

from tombstone.lineage.store import LineageStore
from tombstone.model.artifacts import ArtifactRef
from tombstone.stores.base import ErasableStore, LogicalProbeResult, ProbeSet
from tombstone.util import fingerprint_bytes, sha256_hex

PROBE_QUERIES = 3
PROBE_WORDS = 8


def derived_queries(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    n = len(words)
    first = " ".join(words[:PROBE_WORDS])
    mid_start = max(0, n // 2 - PROBE_WORDS // 2)
    middle = " ".join(words[mid_start : mid_start + PROBE_WORDS])
    last = " ".join(words[-PROBE_WORDS:])
    out: list[str] = []
    for q in (first, middle, last):
        if q and q not in out:
            out.append(q)
    return out


def record_probes(
    lineage: LineageStore,
    artifact_id: str,
    model: str,
    text: str,
    embed: Callable[[Sequence[str]], list[list[float]]],
) -> int:
    """Embed the derived queries and store (hash, embedding) rows for ``artifact_id``."""
    qs = derived_queries(text)
    if not qs:
        return 0
    vecs = embed(qs)
    rows = []
    for q, v in zip(qs, vecs, strict=True):
        rows.append((sha256_hex(q), struct.pack(f"<{len(v)}f", *[float(x) for x in v]).hex()))
    lineage.put_probes(artifact_id, model, rows)
    return len(rows)


def padded_fingerprint(fingerprint_hex: str, dims: int) -> tuple[float, ...]:
    raw = fingerprint_bytes(fingerprint_hex)
    n = len(raw) // 4
    head = struct.unpack(f"<{n}f", raw)
    return tuple(list(head) + [0.0] * max(0, dims - n))[:dims]


def build_probe_set(lineage: LineageStore, ref: ArtifactRef, dims: int, k: int = 40) -> ProbeSet:
    vectors: list[tuple[float, ...]] = []
    if ref.embedding_fingerprint and dims > 0:
        vectors.append(padded_fingerprint(ref.embedding_fingerprint, dims))
    for _qh, _model, emb_hex in lineage.probes(ref.artifact_id):
        raw = bytes.fromhex(emb_hex)
        vec = struct.unpack(f"<{len(raw) // 4}f", raw)
        if dims <= 0 or len(vec) == dims:
            vectors.append(tuple(vec))
    return ProbeSet(artifact_id=ref.artifact_id, vectors=tuple(vectors), k=k)


def run_logical(store: ErasableStore, ref: ArtifactRef, probes: ProbeSet) -> LogicalProbeResult:
    return store.probe_logical(ref, probes)
