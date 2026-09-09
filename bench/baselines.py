"""6.2 — the baselines, implemented faithfully.

Store side (given a backend adapter and the target subject's EMBED refs):
  B0  native delete()                          the store's own delete, nothing else
  B1  delete + vendor "optimize/compact" call  pgvector: VACUUM (plain); qdrant server: optimizer
                                               trigger; chroma / faiss / qdrant-local: NO vendor
                                               compaction API exists → recorded as "n/a (= B0)"
  B2  delete + full index rebuild              re-create the index from the surviving vectors
                                               through the vendor's own API only (no Tombstone
                                               purge/VACUUM of side files)
  B3  Tombstone suppress only                  the saga with reclaim disabled
  B4  Tombstone full                           suppress → reclaim → verify

Model side (bench/unlearn/run_unlearn.py):
  M0  nothing   M1  NPO   M2  gradient difference   M3  Tombstone exact shard retrain
  M4  full retrain from scratch without the subject (the oracle)

A strawman invalidates the comparison: B2 really rebuilds and M4 really retrains.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tombstone.model.artifacts import ArtifactRef
from tombstone.stores._vector import VectorBackendBase

STORE_BASELINES = ("B0", "B1", "B2", "B3", "B4")
MODEL_BASELINES = ("M0", "M1", "M2", "M3", "M4")


def b0_native_delete(store: VectorBackendBase, refs: Sequence[ArtifactRef]) -> str:
    return store.native_delete([r.store_key for r in refs])


def b1_delete_and_vendor_compact(store: VectorBackendBase, refs: Sequence[ArtifactRef]) -> str:
    what = b0_native_delete(store, refs)
    kind = store.kind
    if kind == "pgvector":
        store._conn.execute(f'VACUUM "{store.table}"')  # type: ignore[attr-defined]
        return what + " + VACUUM"
    if kind == "qdrant" and getattr(store, "path", None) is None:
        store._client.update_collection(  # type: ignore[attr-defined]
            store.collection,  # type: ignore[attr-defined]
            optimizers_config=store._models.OptimizersConfigDiff(
                deleted_threshold=0.0, vacuum_min_vector_number=1
            ),  # type: ignore[attr-defined]
        )
        return what + " + optimizer trigger"
    return what + " + n/a (no vendor compaction API for this backend; = B0)"


def b2_delete_and_rebuild(store: VectorBackendBase, refs: Sequence[ArtifactRef]) -> str:
    """Vendor-only rebuild: dump survivors through the public API, drop, re-create, re-add."""
    b0_native_delete(store, refs)
    kind = store.kind
    if kind == "chroma":
        survivors = store._dump_all()
        meta = dict(store._coll.metadata or {})
        store._client.delete_collection(store.collection_name)
        store._coll = store._client.get_or_create_collection(
            store.collection_name,
            metadata=meta or {"hnsw:space": "cosine"},
            embedding_function=None,
        )
        for i in range(0, len(survivors["ids"]), 500):
            sl = slice(i, i + 500)
            if not survivors["ids"][sl]:
                break
            store._coll.upsert(
                ids=survivors["ids"][sl],
                embeddings=survivors["embeddings"][sl],
                metadatas=survivors["metadatas"][sl],
                documents=survivors["documents"][sl],
            )
        return "delete + delete_collection + re-add survivors (no sqlite purge/VACUUM)"
    if kind == "faiss":
        # FAISS has no in-place rebuild: this IS what an app does — new index from survivors.
        survivors = list(store._meta.items())
        new_index = store._new_index(store.dims)
        if survivors:
            import numpy as np

            vecs = [store._index.reconstruct(int(rec["id"])) for _k, rec in survivors]
            new_index.add_with_ids(
                np.asarray(vecs, dtype="float32"),
                np.asarray([int(rec["id"]) for _k, rec in survivors], dtype="int64"),
            )
        store._index = new_index
        store._excluded = set()
        store._persist()
        return "drop mapping + rebuild index from survivors (same as B4 for FAISS)"
    if kind == "qdrant":
        survivors = store._dump_all()
        store._client.delete_collection(store.collection)
        store._client.create_collection(
            store.collection,
            vectors_config=store._models.VectorParams(
                size=store.dims, distance=store._models.Distance.COSINE
            ),
        )
        for i in range(0, len(survivors), 256):
            store._client.upsert(store.collection, points=survivors[i : i + 256], wait=True)
        return "delete + recreate collection + re-add survivors (no sqlite VACUUM)"
    if kind == "pgvector":
        store._conn.execute(f'REINDEX INDEX "{store.index_name}"')
        return "DELETE + REINDEX INDEX (no VACUUM)"
    raise ValueError(kind)


def run_store_baseline(name: str, store: VectorBackendBase, refs: Sequence[ArtifactRef]) -> str:
    if name == "B0":
        return b0_native_delete(store, refs)
    if name == "B1":
        return b1_delete_and_vendor_compact(store, refs)
    if name == "B2":
        return b2_delete_and_rebuild(store, refs)
    raise ValueError(f"{name} is run through the saga, not here")


def describe(name: str) -> dict[str, Any]:
    return {
        "B0": {"label": "native delete()", "method": "store.delete(ids)"},
        "B1": {
            "label": "delete + vendor compact",
            "method": "delete + VACUUM / optimizer where the vendor offers one",
        },
        "B2": {
            "label": "delete + full rebuild",
            "method": "delete + recreate index from survivors via vendor API",
        },
        "B3": {"label": "Tombstone suppress only", "method": "saga, reclaim disabled"},
        "B4": {"label": "Tombstone full", "method": "saga: suppress → reclaim → verify"},
        "M0": {"label": "nothing", "method": "serving model unchanged"},
        "M1": {
            "label": "NPO",
            "method": "negative preference optimisation on the unsharded adapter",
        },
        "M2": {
            "label": "gradient difference",
            "method": "ascent on forget + descent on retain, unsharded adapter",
        },
        "M3": {
            "label": "Tombstone exact shard retrain",
            "method": "drop rows, retrain the subject's shard, recompose",
        },
        "M4": {
            "label": "full retrain (oracle)",
            "method": "retrain the unsharded adapter from scratch without the subject",
        },
    }[name]
