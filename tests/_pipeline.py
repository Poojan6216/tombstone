"""A reusable end-to-end pipeline over the fixture corpus for erase/verify tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests import _corpus

CONFIG = """\
version: 1
scope: default
lineage: {{ backend: sqlite, path: {root}/.tombstone/lineage.db }}
stores:
  - {{ name: "docs", kind: docstore, path: {root}/docs.sqlite }}
{stores}
  - {{ name: "exact-cache", kind: cache_exact, path: {root}/.langchain.db }}
  - {{ name: "semantic-cache", kind: cache_semantic, backing: "{semantic_backing}" }}
  - {{ name: "ft-dataset", kind: dataset, manifest: {root}/train/manifest.json }}
erase:
  require_confirm: true
  reclaim_timeout_s: 60
  semantic_probe_budget: 5
  lock_timeout_s: 20
out_of_scope:
  - "database backups and snapshots"
  - "write-ahead logs and replicas"
  - "embedding-provider-side request logs"
"""

STORE_LINES = {
    "chroma": '  - {{ name: "chroma:kb-v2", kind: chroma, path: {root}/chroma, collection: tombstone-kb, embedding: hash-embed-64 }}',
    "faiss": '  - {{ name: "faiss:kb-v1", kind: faiss, path: {root}/faiss/kb.index, embedding: hash-embed-64 }}',
    "qdrant": '  - {{ name: "qdrant:kb", kind: qdrant, path: {root}/qdrant, collection: kb, embedding: hash-embed-64 }}',
    "pgvector": '  - {{ name: "pgvector:kb-v1", kind: pgvector, dsn: "{dsn}", table: documents, embedding: hash-embed-64 }}',
}


def write_config(root: Path, backends: list[str], pg_dsn: str | None = None) -> Path:
    stores = "\n".join(STORE_LINES[b].format(root=root, dsn=pg_dsn or "") for b in backends)
    backing = {
        "chroma": "chroma:kb-v2",
        "faiss": "faiss:kb-v1",
        "qdrant": "qdrant:kb",
        "pgvector": "pgvector:kb-v1",
    }[backends[0]]
    cfg_path = root / "tombstone.yaml"
    cfg_path.write_text(CONFIG.format(root=root, stores=stores, semantic_backing=backing))
    return cfg_path


def build(
    root: Path,
    backends: list[str],
    pg_dsn: str | None = None,
    rag_subjects: tuple[str, ...] = ("S-0003", "S-0005"),
    shards: int = 4,
) -> dict[str, Any]:
    """init → stamp → docstore → index() into every backend → RAG + caches → dataset."""
    from langchain_core.documents import Document
    from langchain_core.indexing import InMemoryRecordManager, index
    from langchain_core.outputs import Generation
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from tombstone.commands.init import run_init
    from tombstone.integrations.langchain import TombstoneVectorStore
    from tombstone.lineage.stamp import K_CHUNK, stamp
    from tombstone.model.artifacts import ArtifactKind, Scope
    from tombstone.registry import Runtime
    from tombstone.train.dataset import build_dataset

    run_init(root)
    cfg_path = write_config(root, backends, pg_dsn)
    rt = Runtime.shared(cfg_path)
    pepper = rt.pepper()
    corpus = _corpus.load_corpus()
    docs = [
        stamp(
            Document(page_content=d.text, metadata={"source": d.source}),
            d.subject,
            d.source,
            "default",
            pepper=pepper,
            mentions=list(d.mentions),
        )
        for d in corpus
    ]
    docstore = rt.store("docs")
    capture = rt.capture()
    for doc in docs:
        src = capture.ensure_source(doc.metadata, doc.page_content)
        docstore.put(src.artifact_id, "source", doc.page_content, doc.metadata)  # type: ignore[attr-defined]
    chunks = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0).split_documents(docs)
    stores = {}
    names = {
        "chroma": "chroma:kb-v2",
        "faiss": "faiss:kb-v1",
        "qdrant": "qdrant:kb",
        "pgvector": "pgvector:kb-v1",
    }
    for b in backends:
        vs = TombstoneVectorStore.from_config(names[b], config=cfg_path)
        rm = InMemoryRecordManager(names[b])
        rm.create_schema()
        index(chunks, rm, vs, cleanup="incremental", source_id_key="source")
        stores[b] = vs
    primary = stores[backends[0]]
    exact = rt.store("exact-cache")
    semantic = rt.store("semantic-cache")
    queries: dict[str, tuple[str, str, str]] = {}
    for subj in rag_subjects:
        can = next(d.canary for d in corpus if d.subject == subj and d.canary)
        q = can.sentence
        retrieved = primary.similarity_search(q, k=3)
        answer = " ".join(r.page_content for r in retrieved)[:300]
        prompt = f"Context:\n{primary.context_block(retrieved)}\nQ: {q}"
        exact.update(prompt, "extractive-llm", [Generation(text=answer)])  # type: ignore[attr-defined]
        semantic.update(q, answer, [r.metadata[K_CHUNK] for r in retrieved])  # type: ignore[attr-defined]
        queries[subj] = (q, prompt, answer)
    chunk_nodes = [
        n for n in rt.lineage.snapshot(Scope("default")).nodes if n.kind is ArtifactKind.CHUNK
    ]
    pairs = []
    for n in chunk_nodes:
        got = docstore.get(n.artifact_id)  # type: ignore[attr-defined]
        assert got is not None
        pairs.append((n, got[0]))
    build_dataset(capture, "ft-dataset", root / "train" / "manifest.json", pairs, shards=shards)
    return {
        "rt": rt,
        "cfg_path": cfg_path,
        "stores": stores,
        "queries": queries,
        "corpus": corpus,
        "pepper": pepper,
    }
