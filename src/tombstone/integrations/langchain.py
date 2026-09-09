"""LangChain integration: a ``VectorStore`` that captures lineage and honours suppression.

    vs = TombstoneVectorStore.from_config("chroma:kb-v2", embeddings)
    index(docs, record_manager, vs, cleanup="incremental", source_id_key="source")

Every read path — ``similarity_search``, ``similarity_search_by_vector``,
``similarity_search_with_score``, ``max_marginal_relevance_search`` and ``raw_query`` — goes
through the backend's suppression-aware ``query``. ``delete()`` performs the store's *native*
delete (what every app does) and records it, so ``tombstone verify --after-native-delete`` can
report what that actually reached.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from tombstone.embeddings import Embedder, LangChainEmbedder
from tombstone.lineage.capture import Capture
from tombstone.lineage.stamp import K_CHUNK, K_EMBED, require_stamped
from tombstone.stores._vector import VectorBackendBase
from tombstone.stores.base import Hit
from tombstone.stores.docstore import SQLiteDocStore
from tombstone.util import content_hash, sha256_hex


class TombstoneRetriever(VectorStoreRetriever):
    """A retriever whose every query honours suppression (the store does it; this is the name)."""


class TombstoneVectorStore(VectorStore):
    def __init__(
        self,
        backend: VectorBackendBase,
        embedder: Embedder,
        capture: Capture,
        docstore: SQLiteDocStore | None = None,
        langchain_embeddings: Embeddings | None = None,
    ) -> None:
        self.backend = backend
        self.embedder = embedder
        self.capture = capture
        self.docstore = docstore
        self._lc_embeddings = langchain_embeddings
        self._tomb_cache: set[str] = set()
        self._tomb_seq = -1
        capture.register(backend.name, backend.kind)

    # --- construction ------------------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        store_name: str,
        embeddings: Embeddings | Embedder | None = None,
        config: str | Path | None = None,
        docstore_name: str | None = None,
    ) -> TombstoneVectorStore:
        from tombstone.registry import Runtime

        rt = Runtime.shared(config)
        sc = rt.store_config(store_name)
        embedder: Embedder
        lc: Embeddings | None = None
        if embeddings is None:
            assert sc.embedding is not None
            embedder = rt.embedder(sc.embedding)
        elif isinstance(embeddings, Embeddings):
            lc = embeddings
            embedder = LangChainEmbedder(embeddings, sc.embedding or "langchain")
        else:
            embedder = embeddings
        backend = rt.store(store_name, dims=embedder.dims)
        assert isinstance(backend, VectorBackendBase)
        ds: SQLiteDocStore | None = None
        ds_cfgs = [s for s in rt.cfg.stores if s.kind == "docstore"]
        if docstore_name or ds_cfgs:
            d = rt.store(docstore_name or ds_cfgs[0].name)
            assert isinstance(d, SQLiteDocStore)
            ds = d
        return cls(backend, embedder, rt.capture(), ds, lc)

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> TombstoneVectorStore:
        raise NotImplementedError(
            "TombstoneVectorStore.from_texts is intentionally unavailable: build the store with "
            "TombstoneVectorStore.from_config(name, embeddings) so lineage is captured, then add_texts()"
        )

    @property
    def embeddings(self) -> Embeddings | None:
        return self._lc_embeddings

    # --- write path --------------------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        mds = [dict(m) for m in (metadatas or [{} for _ in texts])]
        if len(mds) != len(texts):
            raise ValueError("metadatas must match texts")
        for i, md in enumerate(mds):
            require_stamped(md, f"document {i}")
        keys = (
            list(ids)
            if ids
            else [self._default_key(t, md) for t, md in zip(texts, mds, strict=True)]
        )
        vectors = self.embedder.embed(texts)
        records = self.capture.prepare_embeds(
            self.backend.name, self.embedder.name, keys, vectors, mds, texts, self.embedder.embed
        )
        self.backend.add(records)
        if self.docstore is not None:
            for r in records:
                self.docstore.put(
                    r.chunk_node.artifact_id,
                    "chunk",
                    r.document or "",
                    {k: v for k, v in r.metadata.items() if k != K_EMBED},
                )
        return keys

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        ids = kwargs.pop("ids", None)
        if ids is None:
            ids = (
                [d.id for d in documents]
                if all(getattr(d, "id", None) for d in documents)
                else None
            )
        return self.add_texts(
            [d.page_content for d in documents], [dict(d.metadata) for d in documents], ids=ids
        )

    @staticmethod
    def _default_key(text: str, md: dict[str, Any]) -> str:
        return sha256_hex(content_hash(text) + str(md.get("tombstone.artifact_id", "")))[:32]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """The store's own delete — the way everyone deletes. Recorded, not erased."""
        if not ids:
            return False
        self.backend.native_delete(ids)
        self.capture.record_native_delete(self.backend.name, ids)
        return True

    # --- read path (all suppression-aware) -----------------------------------------------------------

    def _tombstoned(self) -> set[str]:
        """The suppression set, cached against the lineage sequence counter.

        Every retrieval consults this, so the uncached form (a join over the whole tombstones
        table) put a linear scan on the query hot path. Any tombstone write takes a sequence
        number, so a one-row counter read is enough to know the cache is still current.
        """
        seq = self.capture.lineage.current_seq()
        if seq != self._tomb_seq:
            self._tomb_cache = set(self.capture.lineage.tombstoned_ids(self.capture.scope))
            self._tomb_seq = seq
        return self._tomb_cache

    def _live(self, hits: Sequence[Hit]) -> list[Hit]:
        dead = self._tombstoned()
        return [h for h in hits if str(h.metadata.get(K_EMBED, "")) not in dead]

    def _to_doc(self, h: Hit) -> Document:
        md = {k: v for k, v in h.metadata.items() if k != "tombstone.suppressed"}
        return Document(page_content=h.document or "", metadata=md, id=h.key)

    def _embed_query(self, query: str) -> list[float]:
        return self.embedder.embed([query])[0]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return self.similarity_search_by_vector(self._embed_query(query), k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return [self._to_doc(h) for h in self._live(self.backend.query(embedding, k + 20))[:k]]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        hits = self._live(self.backend.query(self._embed_query(query), k + 20))[:k]
        return [(self._to_doc(h), h.score) for h in hits]

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any
    ) -> list[Document]:
        return self.max_marginal_relevance_search_by_vector(
            self._embed_query(query), k, fetch_k, lambda_mult
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        hits = self._live(self.backend.query_mmr(embedding, k + 20, fetch_k + 20, lambda_mult))[:k]
        return [self._to_doc(h) for h in hits]

    def raw_query(self, embedding: Sequence[float], k: int = 4) -> list[Hit]:
        """The lowest-level query the wrapper exposes. Still suppression-aware."""
        return self._live(self.backend.query(embedding, k + 20))[:k]

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        hits = self.backend.get(list(ids))
        live = {h.key for h in self._live(list(hits.values()))}
        return [self._to_doc(hits[i]) for i in ids if i in live]

    def as_retriever(self, **kwargs: Any) -> TombstoneRetriever:
        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return TombstoneRetriever(vectorstore=self, tags=tags, **kwargs)

    # --- helpers for RAG apps ----------------------------------------------------------------------

    @staticmethod
    def context_block(docs: Sequence[Document]) -> str:
        """Render retrieved chunks for a prompt with a hidden delimiter listing chunk ids, so the
        exact cache can record parent edges (see stores/cache_exact.py)."""
        ids = [str(d.metadata.get(K_CHUNK, "")) for d in docs if d.metadata.get(K_CHUNK)]
        body = "\n\n".join(d.page_content for d in docs)
        return f"{body}\n<!-- tombstone:chunks={','.join(ids)} -->"
