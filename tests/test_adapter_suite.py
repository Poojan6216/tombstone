"""8.5: the shared adapter test suite every ErasableStore must pass, run here against the
in-memory dict store written from docs/adapters.md alone, and against the docstore."""

from __future__ import annotations

from pathlib import Path

import pytest

from tombstone.model.artifacts import ArtifactKind, ArtifactRef, Scope
from tombstone.model.status import VerifyLevel
from tombstone.stores.base import ErasableStore, ProbeSet
from tombstone.stores.docstore import SQLiteDocStore
from tombstone.stores.memory import MemoryStore


def _ref(store: str, key: str) -> ArtifactRef:
    return ArtifactRef(
        f"01{key}", ArtifactKind.SOURCE, store, key, Scope("default"), "h" * 64, None
    )


def adapter_contract(store: ErasableStore, put: object, refs: list[ArtifactRef]) -> None:
    """The contract from docs/adapters.md. ``put`` writes one record for a ref."""
    assert isinstance(store, ErasableStore)
    assert store.name and store.kind and store.version()
    assert VerifyLevel.LOGICAL in store.capabilities
    for r in refs:
        put(r)  # type: ignore[operator]
    assert store.count() == len(refs)
    assert set(store.sample_keys(10)) == {r.store_key for r in refs}
    ps = ProbeSet(refs[0].artifact_id)
    # power: a present record is found
    assert store.probe_logical(refs[0], ps).found
    if VerifyLevel.PHYSICAL in store.capabilities:
        assert store.probe_physical(refs[0]).found
    # suppress hides without removing bytes; idempotent
    store.suppress(refs[:1])
    store.suppress(refs[:1])
    assert not store.probe_logical(refs[0], ps).found
    assert store.probe_logical(refs[1], ps).found
    if VerifyLevel.PHYSICAL in store.capabilities:
        assert store.probe_physical(refs[0]).found
    # reclaim removes bytes; second reclaim is a no-op
    r1 = store.reclaim(refs[:1])
    assert not r1.noop and r1.method
    if VerifyLevel.PHYSICAL in store.capabilities:
        assert not store.probe_physical(refs[0]).found
    assert store.reclaim(refs[:1]).noop
    assert store.count() == len(refs) - 1
    assert store.probe_logical(refs[1], ps).found
    store.close()


def test_memory_store_passes_contract() -> None:
    store = MemoryStore("mem")
    refs = [_ref("mem", f"k{i}") for i in range(3)]
    adapter_contract(store, lambda r: store.put(r.store_key, {"artifact_id": r.artifact_id}), refs)


def test_docstore_passes_contract(tmp_path: Path) -> None:
    store = SQLiteDocStore("docs", tmp_path / "d.sqlite")
    refs = [_ref("docs", f"k{i}") for i in range(3)]
    adapter_contract(
        store,
        lambda r: store.put(
            r.store_key, "source", "text " + r.store_key, {"artifact_id": r.artifact_id}
        ),
        refs,
    )


@pytest.mark.parametrize("backend", ["faiss", "chroma", "qdrant", "pgvector"])
def test_vector_backends_pass_contract(
    backend: str, tmp_path: Path, pepper: bytes, request: pytest.FixtureRequest
) -> None:
    from tests import _stores

    _stores.skip_unless(backend)
    dsn = request.getfixturevalue("pg_database") if backend == "pgvector" else None
    store = _stores.make_backend(backend, tmp_path, pg_dsn=dsn)
    emb = _stores.embedder()
    docs = _stores.stamped_docs(pepper, n_subjects=3, per_subject=1)[:3]
    with _stores.lineage_and_capture(tmp_path) as (_lineage, capture):
        texts = [x for x, _ in docs]
        recs = capture.prepare_embeds(
            store.name,
            emb.name,
            [f"k{i}" for i in range(3)],
            emb.embed(texts),
            [m for _, m in docs],
            texts,
        )
        refs = [r.embed_node.ref() for r in recs]
        pending = {r.key: r for r in recs}
        adapter_contract(store, lambda r: store.add([pending[r.store_key]]), refs)
