"""1.2: 50 stamped chunks into all four backends → 200 EMBED nodes with fingerprints and edges."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import _stores
from tombstone.lineage.stamp import K_EMBED
from tombstone.model.artifacts import ArtifactKind, Scope, SubjectRef
from tombstone.util import fingerprint_f32


@pytest.mark.parametrize("backend", _stores.BACKENDS)
def test_capture_creates_embed_nodes_with_fingerprints(
    backend: str, tmp_path: Path, pepper: bytes, request: pytest.FixtureRequest
) -> None:
    dsn = request.getfixturevalue("pg_database") if backend == "pgvector" else None
    store = _stores.make_backend(backend, tmp_path, pg_dsn=dsn)
    emb = _stores.embedder()
    docs = _stores.stamped_docs(pepper, n_subjects=5, per_subject=2)[:50]
    with _stores.lineage_and_capture(tmp_path) as (lineage, capture):
        capture.register(store.name, store.kind)
        texts = [t for t, _ in docs]
        mds = [m for _, m in docs]
        keys = [f"k{i}" for i in range(len(docs))]
        vectors = emb.embed(texts)
        records = capture.prepare_embeds(store.name, emb.name, keys, vectors, mds, texts)
        store.add(records)
        assert store.count() == 50
        embeds = [
            n for n in lineage.snapshot(Scope("default")).nodes if n.kind is ArtifactKind.EMBED
        ]
        assert len(embeds) == 50
        by_key = {n.store_key: n for n in embeds}
        for key, vec in zip(keys, vectors, strict=True):
            assert by_key[key].embedding_fingerprint == fingerprint_f32(vec)
            assert by_key[key].store == store.name
        # every EMBED has exactly one chunk parent, tagged with the model name
        for n in embeds:
            parents = lineage.edges_to([n.artifact_id])
            assert len(parents) == 1 and parents[0].via == f"embed:{emb.name}"
            chunk = lineage.node(parents[0].parent)
            assert chunk is not None and chunk.kind is ArtifactKind.CHUNK
            src = lineage.edges_to([chunk.artifact_id])
            assert any(e.via == "chunk" for e in src)
        # stored metadata carries the embed id; id lookup and filter both work
        hit = store.get([keys[0]])[keys[0]]
        assert hit.metadata[K_EMBED] == by_key[keys[0]].artifact_id
        assert [h.key for h in store.filter(K_EMBED, by_key[keys[3]].artifact_id)] == [keys[3]]
        # top-k with the chunk's own vector finds itself
        top = store.query(vectors[7], 5)
        assert top and top[0].key == keys[7]
        # the stored vector equals the fingerprint bytes (float32 round trip)
        v = store._vector_of(keys[7])
        assert v is not None and fingerprint_f32(v) == by_key[keys[7]].embedding_fingerprint
    store.close()


def test_multi_index_yields_one_chunk_two_embeds(tmp_path: Path, pepper: bytes) -> None:
    a = _stores.make_backend("faiss", tmp_path / "a", name="faiss:a")
    b = _stores.make_backend("faiss", tmp_path / "b", name="faiss:b")
    emb = _stores.embedder()
    docs = _stores.stamped_docs(pepper, n_subjects=2, per_subject=1)[:10]
    with _stores.lineage_and_capture(tmp_path) as (lineage, capture):
        texts = [t for t, _ in docs]
        mds = [m for _, m in docs]
        vectors = emb.embed(texts)
        ra = capture.prepare_embeds(
            a.name, emb.name, [f"a{i}" for i in range(10)], vectors, mds, texts
        )
        rb = capture.prepare_embeds(
            b.name, "other-model", [f"b{i}" for i in range(10)], vectors, mds, texts
        )
        a.add(ra)
        b.add(rb)
        snap = lineage.snapshot(Scope("default"))
        kinds = lineage.counts_by_kind(Scope("default"))
        assert kinds["chunk"] == 10 and kinds["embed"] == 20 and kinds["source"] == 2
        for x, y in zip(ra, rb, strict=True):
            assert x.chunk_node.artifact_id == y.chunk_node.artifact_id
            assert x.embed_node.artifact_id != y.embed_node.artifact_id
        vias = {e.via for e in snap.edges}
        assert {"chunk", f"embed:{emb.name}", "embed:other-model"} <= vias
    a.close()
    b.close()


def test_capture_is_idempotent_for_same_key(tmp_path: Path, pepper: bytes) -> None:
    store = _stores.make_backend("faiss", tmp_path)
    emb = _stores.embedder()
    docs = _stores.stamped_docs(pepper, n_subjects=1, per_subject=1)[:3]
    with _stores.lineage_and_capture(tmp_path) as (lineage, capture):
        texts = [t for t, _ in docs]
        mds = [m for _, m in docs]
        vectors = emb.embed(texts)
        r1 = capture.prepare_embeds(store.name, emb.name, ["k0", "k1", "k2"], vectors, mds, texts)
        r2 = capture.prepare_embeds(store.name, emb.name, ["k0", "k1", "k2"], vectors, mds, texts)
        assert [r.embed_node.artifact_id for r in r1] == [r.embed_node.artifact_id for r in r2]
        assert lineage.counts_by_kind(Scope("default"))["embed"] == 3
    store.close()


def test_native_delete_is_recorded(tmp_path: Path, pepper: bytes) -> None:
    store = _stores.make_backend("faiss", tmp_path)
    emb = _stores.embedder()
    docs = _stores.stamped_docs(pepper, n_subjects=1, per_subject=1)[:2]
    with _stores.lineage_and_capture(tmp_path) as (lineage, capture):
        texts = [t for t, _ in docs]
        recs = capture.prepare_embeds(
            store.name, emb.name, ["k0", "k1"], emb.embed(texts), [m for _, m in docs], texts
        )
        store.add(recs)
        store.native_delete(["k0"])
        marked = capture.record_native_delete(store.name, ["k0"])
        assert marked == [recs[0].embed_node.artifact_id]
        info = lineage.tombstone_info(marked[0])
        assert info is not None and info[2] == "native-delete"
        assert "k0" not in store.get(["k0", "k1"])
    store.close()


def test_scope_mismatch_refused(tmp_path: Path, pepper: bytes) -> None:
    docs = _stores.stamped_docs(pepper, n_subjects=1, per_subject=1, scope="tenant-b")[:1]
    with _stores.lineage_and_capture(tmp_path, scope="tenant-a") as (_lineage, capture):
        with pytest.raises(ValueError, match="scope"):
            capture.ensure_source(docs[0][1])


def test_subject_ref_never_raw(tmp_path: Path, pepper: bytes) -> None:
    docs = _stores.stamped_docs(pepper, n_subjects=1, per_subject=1)[:1]
    with _stores.lineage_and_capture(tmp_path) as (lineage, capture):
        node = capture.ensure_source(docs[0][1], "text")
        assert node.subject_hmac == SubjectRef.from_raw("S-0000", pepper).hmac
        assert "file-0-0" not in node.store_key
