"""Phase 1 gate: fixture corpus → LangChain index() → two vector stores → a RAG call with
caching → dataset build. `trace S-0003` returns every artifact with correct edges and empty gaps.
Then bypass capture for one store and assert gaps is non-empty."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import _corpus, _stores
from tests.conftest import requires_chroma, requires_faiss, requires_langchain
from tombstone.lineage.stamp import K_CHUNK, stamp
from tombstone.model.artifacts import ArtifactKind, Scope
from tombstone.util import content_hash

pytestmark = [requires_langchain, requires_chroma, requires_faiss]

CONFIG = """\
version: 1
scope: default
lineage: {{ backend: sqlite, path: {root}/.tombstone/lineage.db }}
stores:
  - {{ name: "docs", kind: docstore, path: {root}/docs.sqlite }}
  - {{ name: "chroma:kb-v2", kind: chroma, path: {root}/chroma, collection: tombstone-kb, embedding: hash-embed-64 }}
  - {{ name: "faiss:kb-v1", kind: faiss, path: {root}/faiss/kb.index, embedding: hash-embed-64 }}
  - {{ name: "exact-cache", kind: cache_exact, path: {root}/.langchain.db }}
  - {{ name: "semantic-cache", kind: cache_semantic, backing: "chroma:kb-v2" }}
  - {{ name: "ft-dataset", kind: dataset, manifest: {root}/train/manifest.json }}
out_of_scope:
  - "database backups and snapshots"
  - "write-ahead logs and replicas"
"""


def build_pipeline(root: Path):
    """Run the whole ingestion pipeline; returns the Runtime (open) and useful handles."""
    from langchain_core.indexing import InMemoryRecordManager, index
    from langchain_core.outputs import Generation
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from tombstone.commands.init import run_init
    from tombstone.integrations.langchain import TombstoneVectorStore
    from tombstone.registry import Runtime
    from tombstone.stores.cache_exact import TombstoneExactCache
    from tombstone.stores.cache_semantic import SemanticCache
    from tombstone.train.dataset import build_dataset

    run_init(root)
    (root / "tombstone.yaml").write_text(CONFIG.format(root=root))
    rt = Runtime.shared(root / "tombstone.yaml")
    pepper = rt.pepper()
    corpus = _corpus.load_corpus()
    docs = []
    from langchain_core.documents import Document

    for d in corpus:
        doc = Document(page_content=d.text, metadata={"source": d.source})
        docs.append(
            stamp(doc, d.subject, d.source, "default", pepper=pepper, mentions=list(d.mentions))
        )
    # source documents into the docstore (SOURCE nodes with content hashes)
    docstore = rt.store("docs")
    capture = rt.capture()
    for doc in docs:
        src = capture.ensure_source(doc.metadata, doc.page_content)
        docstore.put(src.artifact_id, "source", doc.page_content, doc.metadata)  # type: ignore[attr-defined]
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    vs_a = TombstoneVectorStore.from_config("chroma:kb-v2", config=root / "tombstone.yaml")
    vs_b = TombstoneVectorStore.from_config("faiss:kb-v1", config=root / "tombstone.yaml")
    rm_a, rm_b = InMemoryRecordManager("a"), InMemoryRecordManager("b")
    rm_a.create_schema()
    rm_b.create_schema()
    index(chunks, rm_a, vs_a, cleanup="incremental", source_id_key="source")
    index(chunks, rm_b, vs_b, cleanup="incremental", source_id_key="source")
    # a RAG call: retrieve, "answer" (extractive stand-in for an LLM), cache both ways
    exact = rt.store("exact-cache")
    assert isinstance(exact, TombstoneExactCache)
    semantic = rt.store("semantic-cache")
    assert isinstance(semantic, SemanticCache)
    queries = {}
    for subj in ("S-0003", "S-0005"):
        can = next(d.canary for d in corpus if d.subject == subj and d.canary)
        q = (
            can.sentence
        )  # the app's question quotes the subject's own sentence → its chunk ranks first
        retrieved = vs_a.similarity_search(q, k=3)
        answer = " ".join(r.page_content for r in retrieved)[:300]
        prompt = f"Context:\n{vs_a.context_block(retrieved)}\nQ: {q}"
        exact.update(prompt, "extractive-llm", [Generation(text=answer)])
        semantic.update(q, answer, [r.metadata[K_CHUNK] for r in retrieved])
        queries[subj] = (q, prompt, answer)
    # dataset from every captured chunk
    lineage = rt.lineage
    chunk_nodes = [
        n for n in lineage.snapshot(Scope("default")).nodes if n.kind is ArtifactKind.CHUNK
    ]
    pairs = []
    for n in chunk_nodes:
        got = docstore.get(n.artifact_id)  # type: ignore[attr-defined]
        assert got is not None
        pairs.append((n, got[0]))
    build_dataset(capture, "ft-dataset", root / "train" / "manifest.json", pairs, shards=4)
    return rt, {"vs_a": vs_a, "vs_b": vs_b, "queries": queries, "corpus": corpus, "pepper": pepper}


def test_phase1_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    rt, h = build_pipeline(tmp_path)
    from tombstone.commands.trace import run_trace

    t, report = run_trace(rt, "S-0003")
    assert t.gaps == (), t.gaps
    kinds = {}
    for a in t.artifacts:
        kinds[a.kind] = kinds.get(a.kind, 0) + 1
    corpus = h["corpus"]
    n_sources = sum(1 for d in corpus if d.subject == "S-0003")
    assert kinds[ArtifactKind.SOURCE] == n_sources
    assert kinds[ArtifactKind.CHUNK] >= n_sources
    assert kinds[ArtifactKind.EMBED] == 2 * kinds[ArtifactKind.CHUNK]  # two indexes
    assert kinds[ArtifactKind.TRAIN] == kinds[ArtifactKind.CHUNK]
    assert kinds.get(ArtifactKind.CACHE, 0) >= 1  # the RAG call retrieved S-0003 chunks
    # every EMBED's parent is one of the traced chunks; every chunk's parent is a traced source
    traced = {a.artifact_id for a in t.artifacts}
    for e in t.edges:
        assert e.parent in traced and e.child in traced
    embed_parents = {e.parent for e in t.edges if e.via.startswith("embed:")}
    assert embed_parents <= {a.artifact_id for a in t.artifacts if a.kind is ArtifactKind.CHUNK}
    # S-0003's doc mentions S-0005 → third-party hit on S-0005's trace, not on ours
    t5, _ = run_trace(rt, "S-0005")
    hits = {a.artifact_id for a in t5.third_party_hits}
    s3_sources = {a.artifact_id for a in t.artifacts if a.kind is ArtifactKind.SOURCE}
    assert hits & s3_sources, (
        "S-0003's document mentions S-0005 and must be listed as a third-party hit"
    )
    assert not (hits & {a.artifact_id for a in t5.artifacts})
    # content hashes match the docstore content (no content in lineage)
    for a in t.artifacts:
        if a.kind is ArtifactKind.SOURCE:
            got = rt.store("docs").get(a.artifact_id)  # type: ignore[attr-defined]
            assert got and content_hash(got[0]) == a.content_hash
    # the trace is persisted and reloadable by id
    assert rt.lineage.load_trace(t.trace_id) == t
    # now bypass capture for one store: 20 docs straight into chroma
    emb = _stores.embedder()
    texts = [f"legacy doc {i}" for i in range(20)]
    h["vs_a"].backend._coll.add(
        ids=[f"legacy{i}" for i in range(20)], embeddings=emb.embed(texts), documents=texts
    )
    t2, _ = run_trace(rt, "S-0003")
    assert t2.gaps and any("chroma:kb-v2" in g for g in t2.gaps)
    assert t2.trace_id != t.trace_id  # the snapshot changed, so the trace id changed
    rt.close()
