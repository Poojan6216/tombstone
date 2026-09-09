"""Shared helpers to build every backend for tests, with a deterministic hash embedder."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tombstone.embeddings import HashEmbedder
from tombstone.lineage.capture import Capture
from tombstone.lineage.store import LineageStore
from tombstone.model.artifacts import Scope
from tombstone.stores._vector import VectorBackendBase

DIMS = 64
BACKENDS = ["chroma", "faiss", "qdrant", "pgvector"]


def has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def skip_unless(backend: str) -> None:
    mod = {
        "chroma": "chromadb",
        "faiss": "faiss",
        "qdrant": "qdrant_client",
        "pgvector": "psycopg",
    }[backend]
    if not has_module(mod):
        pytest.skip(f"{mod} not installed")


def make_backend(
    backend: str, tmp: Path, name: str | None = None, pg_dsn: str | None = None, dims: int = DIMS
) -> VectorBackendBase:
    skip_unless(backend)
    name = name or f"{backend}:kb"
    if backend == "chroma":
        from tombstone.stores.chroma import ChromaStore

        return ChromaStore(
            name,
            tmp / "chroma",
            collection="tombstone-kb",
            embedding_model="hash-embed-64",
            dims=dims,
        )
    if backend == "faiss":
        from tombstone.stores.faiss import FaissStore

        return FaissStore(
            name, tmp / "faiss" / "kb.index", embedding_model="hash-embed-64", dims=dims
        )
    if backend == "qdrant":
        from tombstone.stores.qdrant import QdrantStore

        return QdrantStore(
            name, path=tmp / "qdrant", collection="kb", embedding_model="hash-embed-64", dims=dims
        )
    if backend == "pgvector":
        if pg_dsn is None:
            pytest.skip("no Postgres")
        from tombstone.stores.pgvector import PgVectorStore

        return PgVectorStore(
            name, pg_dsn, table="documents", embedding_model="hash-embed-64", dims=dims
        )
    raise ValueError(backend)


@contextmanager
def lineage_and_capture(
    tmp: Path, scope: str = "default"
) -> Iterator[tuple[LineageStore, Capture]]:
    store = LineageStore.open_sqlite(tmp / "lineage.db")
    try:
        yield store, Capture(store, Scope(scope))
    finally:
        store.close()


def embedder() -> HashEmbedder:
    return HashEmbedder(DIMS)


def stamped_docs(
    pepper: bytes, n_subjects: int = 5, per_subject: int = 2, scope: str = "default"
) -> list[tuple[str, dict[str, Any]]]:
    """(text, stamped metadata) pairs, several chunks per source."""
    from tombstone.lineage.stamp import stamp

    out: list[tuple[str, dict[str, Any]]] = []
    for s in range(n_subjects):
        for d in range(per_subject):
            md = stamp(
                {"source": f"file-{s}-{d}.txt"},
                f"S-{s:04d}",
                f"file-{s}-{d}.txt",
                scope,
                pepper=pepper,
            )
            for c in range(5):
                out.append(
                    (
                        f"subject {s} document {d} chunk {c} says something unique {s * 100 + d * 10 + c}",
                        dict(md),
                    )
                )
    return out
