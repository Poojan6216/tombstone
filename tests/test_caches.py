"""1.4: exact cache (LangChain SQLiteCache) and the in-repo semantic cache capture lineage."""

from __future__ import annotations

from pathlib import Path

from tests import _stores
from tests.conftest import requires_langchain
from tombstone.lineage.capture import Capture
from tombstone.lineage.stamp import K_CHUNK
from tombstone.lineage.store import LineageStore
from tombstone.model.artifacts import ArtifactKind, Scope


def _chunks(tmp_path: Path, pepper: bytes):
    """Three chunks of one subject captured in a FAISS index; returns (capture, chunk ids, store)."""
    store = _stores.make_backend("faiss", tmp_path)
    emb = _stores.embedder()
    docs = _stores.stamped_docs(pepper, n_subjects=1, per_subject=1)[:3]
    lineage = LineageStore.open_sqlite(tmp_path / "lineage.db")
    capture = Capture(lineage, Scope("default"))
    texts = [t for t, _ in docs]
    recs = capture.prepare_embeds(
        store.name, emb.name, ["k0", "k1", "k2"], emb.embed(texts), [m for _, m in docs], texts
    )
    store.add(recs)
    return lineage, capture, [r.chunk_node.artifact_id for r in recs], store, emb


@requires_langchain
def test_exact_cache_records_parent_edges(tmp_path: Path, pepper: bytes) -> None:
    from langchain_core.outputs import Generation

    from tombstone.stores.cache_exact import TombstoneExactCache, cache_key

    lineage, capture, chunk_ids, store, _ = _chunks(tmp_path, pepper)
    cache = TombstoneExactCache("exact-cache", tmp_path / ".langchain.db", capture)
    prompt = (
        "Answer using:\nchunk text...\n<!-- tombstone:chunks="
        + ",".join(chunk_ids)
        + " -->\nQ: what?"
    )
    cache.update(prompt, "fake-llm", [Generation(text="the answer quotes chunk 0")])
    assert cache.lookup(prompt, "fake-llm") is not None
    nodes = [n for n in lineage.snapshot(Scope("default")).nodes if n.kind is ArtifactKind.CACHE]
    assert len(nodes) == 1
    parents = lineage.edges_to([nodes[0].artifact_id])
    assert {e.parent for e in parents} == set(chunk_ids) and all(
        e.via == "cache:exact" for e in parents
    )
    assert nodes[0].store_key.startswith(cache_key(prompt, "fake-llm"))
    assert cache.count() == 1
    # suppression deletes immediately (caches have no hide state)
    cache.suppress([nodes[0].ref()])
    assert cache.lookup(prompt, "fake-llm") is None
    assert not cache.probe_logical(nodes[0].ref(), _probe(nodes[0].artifact_id)).found
    store.close()
    lineage.close()


def _probe(aid: str):
    from tombstone.stores.base import ProbeSet

    return ProbeSet(artifact_id=aid)


def test_semantic_cache_hit_on_paraphrase_returns_same_node(tmp_path: Path, pepper: bytes) -> None:
    from tombstone.stores.cache_semantic import SemanticCache

    lineage, capture, chunk_ids, store, emb = _chunks(tmp_path, pepper)
    backing = _stores.make_backend("faiss", tmp_path / "cache", name="faiss:kb#semantic-cache")
    cache = SemanticCache("semantic-cache", backing, emb, capture, threshold=0.5)
    q = "subject 0 document 0 chunk 0 says something unique 0"
    nodes = cache.update(q, "answer built from chunk 0 1 2", chunk_ids)
    assert len(nodes) == 1
    parents = lineage.edges_to([nodes[0].artifact_id])
    assert {e.parent for e in parents} == set(chunk_ids) and all(
        e.via == "cache:semantic" for e in parents
    )
    assert nodes[0].embedding_fingerprint is not None
    hit = cache.lookup("subject 0 document 0 chunk 0 says something unique 0 please")  # paraphrase
    assert hit is not None and hit[0].startswith("answer built")
    assert hit[1][K_CHUNK].split(",") == chunk_ids
    assert cache.lookup("completely unrelated weather forecast for tomorrow morning") is None
    cache.suppress([nodes[0].ref()])
    assert cache.lookup(q) is None
    store.close()
    backing.close()
    lineage.close()


@requires_langchain
def test_exact_cache_parses_delimiter_edge_cases() -> None:
    from tombstone.stores.cache_exact import parse_chunk_ids

    assert parse_chunk_ids("no delimiter") == []
    assert parse_chunk_ids("<!-- tombstone:chunks= -->") == []
    assert parse_chunk_ids("x <!-- tombstone:chunks=B,A --> y <!-- tombstone:chunks=A,C -->") == [
        "A",
        "B",
        "C",
    ]
