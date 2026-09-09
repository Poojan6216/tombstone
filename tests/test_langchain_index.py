"""1.3: LangChain index() with cleanup="incremental" over the fixture corpus, twice, with 5
modified documents between runs; lineage must match the RecordManager's view exactly."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import _corpus, _stores
from tests.conftest import requires_langchain
from tombstone.lineage.capture import Capture
from tombstone.lineage.stamp import K_ARTIFACT, K_CHUNK, stamp
from tombstone.lineage.store import LineageStore
from tombstone.model.artifacts import ArtifactKind, Scope

pytestmark = requires_langchain


def _stamped_documents(pepper: bytes, modify: set[str] = frozenset()) -> list:
    from langchain_core.documents import Document

    docs = []
    for d in _corpus.load_corpus():
        text = d.text + (" [revised]" if d.doc_id in modify else "")
        doc = Document(page_content=text, metadata={"source": d.source})
        docs.append(
            stamp(doc, d.subject, d.source, "default", pepper=pepper, mentions=list(d.mentions))
        )
    return docs


def _split(docs: list) -> list:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=160, chunk_overlap=0)
    return splitter.split_documents(docs)


def _build_vs(tmp_path: Path):
    from tombstone.integrations.langchain import TombstoneVectorStore
    from tombstone.stores.docstore import SQLiteDocStore

    lineage_store = LineageStore.open_sqlite(tmp_path / "lineage.db")
    capture = Capture(lineage_store, Scope("default"))
    backend = _stores.make_backend("faiss", tmp_path)
    docstore = SQLiteDocStore("docs", tmp_path / "docs.sqlite")
    vs = TombstoneVectorStore(backend, _stores.embedder(), capture, docstore)
    return vs, lineage_store


def test_index_incremental_twice_matches_record_manager(tmp_path: Path, pepper: bytes) -> None:
    from langchain_core.indexing import InMemoryRecordManager, index

    vs, lineage = _build_vs(tmp_path)
    rm = InMemoryRecordManager(namespace="test")
    rm.create_schema()

    first = index(
        _split(_stamped_documents(pepper)), rm, vs, cleanup="incremental", source_id_key="source"
    )
    assert first["num_added"] > 0 and first["num_deleted"] == 0
    keys_after_first = set(rm.list_keys())
    lineage_keys = {
        n.store_key
        for n in lineage.snapshot(Scope("default")).nodes
        if n.kind is ArtifactKind.EMBED
    }
    assert lineage_keys == keys_after_first

    # Stable stamps: re-indexing unchanged docs must add nothing (otherwise index() is not incremental).
    again = index(
        _split(_stamped_documents(pepper)), rm, vs, cleanup="incremental", source_id_key="source"
    )
    assert again["num_added"] == 0 and again["num_deleted"] == 0 and again["num_skipped"] > 0

    modified = {"S-0001-0", "S-0002-1", "S-0003-0", "S-0004-2", "S-0005-1"}
    second = index(
        _split(_stamped_documents(pepper, modified)),
        rm,
        vs,
        cleanup="incremental",
        source_id_key="source",
    )
    assert second["num_added"] >= 5 and second["num_deleted"] >= 5
    keys_after_second = set(rm.list_keys())
    snap = lineage.snapshot(Scope("default"))
    live = {
        n.store_key
        for n in snap.nodes
        if n.kind is ArtifactKind.EMBED and n.artifact_id not in snap.tombstoned
    }
    dead = {
        n.store_key
        for n in snap.nodes
        if n.kind is ArtifactKind.EMBED and n.artifact_id in snap.tombstoned
    }
    assert live == keys_after_second
    assert dead == keys_after_first - keys_after_second
    for n in snap.nodes:
        if n.artifact_id in snap.tombstoned:
            info = lineage.tombstone_info(n.artifact_id)
            assert info is not None and info[2] == "native-delete"
    # new chunks are linked to their sources
    for n in snap.nodes:
        if n.kind is ArtifactKind.EMBED and n.artifact_id not in snap.tombstoned:
            parents = lineage.edges_to([n.artifact_id])
            assert parents and lineage.node(parents[0].parent).kind is ArtifactKind.CHUNK
    # the store agrees with the record manager on what is live
    assert set(vs.backend.all_keys()) == keys_after_second
    vs.backend.close()
    lineage.close()


def test_retriever_paths_and_context_block(tmp_path: Path, pepper: bytes) -> None:
    from langchain_core.indexing import InMemoryRecordManager, index

    vs, lineage = _build_vs(tmp_path)
    rm = InMemoryRecordManager(namespace="t")
    rm.create_schema()
    index(_split(_stamped_documents(pepper)), rm, vs, cleanup="incremental", source_id_key="source")
    q = "Support transcript 1-0 refund request"
    docs = vs.similarity_search(q, k=3)
    assert len(docs) == 3 and all(K_CHUNK in d.metadata and K_ARTIFACT in d.metadata for d in docs)
    by_vec = vs.similarity_search_by_vector(vs.embedder.embed([q])[0], k=3)
    assert [d.id for d in by_vec] == [d.id for d in docs]
    scored = vs.similarity_search_with_score(q, k=2)
    assert len(scored) == 2
    mmr = vs.max_marginal_relevance_search(q, k=3, fetch_k=10)
    assert len(mmr) == 3
    retr = vs.as_retriever(search_kwargs={"k": 2})
    assert len(retr.invoke(q)) == 2
    block = vs.context_block(docs)
    assert "<!-- tombstone:chunks=" in block and docs[0].metadata[K_CHUNK] in block
    with pytest.raises(NotImplementedError):
        vs.from_texts(["x"], vs.embedder)  # type: ignore[arg-type]
    vs.backend.close()
    lineage.close()


def test_unstamped_documents_are_refused(tmp_path: Path) -> None:
    from langchain_core.documents import Document

    vs, lineage = _build_vs(tmp_path)
    with pytest.raises(ValueError, match="not stamped"):
        vs.add_documents([Document(page_content="hi", metadata={"source": "x"})])
    vs.backend.close()
    lineage.close()
