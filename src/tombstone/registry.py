"""Build store adapters and embedders from ``tombstone.yaml``. The one place config meets code."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tombstone.config import Installation, StoreConfig, TombstoneConfig, load_config
from tombstone.embeddings import Embedder, get_embedder
from tombstone.errors import ConfigError
from tombstone.lineage.capture import Capture
from tombstone.lineage.store import LineageStore
from tombstone.model.artifacts import Scope
from tombstone.stores.base import ErasableStore

_SHARED: dict[str, Runtime] = {}


def _abs(base: Path, p: str | None) -> Path:
    assert p is not None
    path = Path(p)
    return path if path.is_absolute() else (base / path)


class Runtime:
    """Config + installation + lineage store + lazily-built store adapters."""

    def __init__(self, cfg: TombstoneConfig, cfg_path: Path) -> None:
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.root = cfg_path.parent
        self.inst = Installation(_abs(self.root, cfg.state_dir))
        self.scope = Scope(cfg.scope)
        self._lineage: LineageStore | None = None
        self._stores: dict[str, ErasableStore] = {}
        self._build_lock = threading.RLock()
        self._embedders: dict[str, Embedder] = {}

    @classmethod
    def load(cls, explicit: str | Path | None = None) -> Runtime:
        cfg, path = load_config(explicit)
        return cls(cfg, path)

    @classmethod
    def shared(cls, explicit: str | Path | None = None) -> Runtime:
        """One Runtime per config path per process, so every entry point (LangChain wrapper,
        trace, erase) reuses the same store adapters. Local-mode backends (Qdrant) allow only
        one client per path; sharing is what makes "wrap the store in one line" safe."""
        cfg, path = load_config(explicit)
        key = str(path.resolve())
        rt = _SHARED.get(key)
        if rt is None:
            rt = cls(cfg, path)
            _SHARED[key] = rt
        return rt

    # --- lineage -----------------------------------------------------------------------------

    @property
    def lineage(self) -> LineageStore:
        if self._lineage is None:
            lc = self.cfg.lineage
            if lc.backend == "sqlite":
                self._lineage = LineageStore.open_sqlite(_abs(self.root, lc.path))
            else:
                self._lineage = LineageStore.open(lc)
        return self._lineage

    def capture(self) -> Capture:
        docstores = [s for s in self.cfg.stores if s.kind == "docstore"]
        source_store = docstores[0].name if docstores else "source"
        chunk_store = docstores[0].name if docstores else "chunks"
        return Capture(self.lineage, self.scope, source_store=source_store, chunk_store=chunk_store)

    def pepper(self) -> bytes:
        return self.inst.read_pepper()

    # --- embedders -----------------------------------------------------------------------------

    def embedder(self, name: str, dims: int = 64) -> Embedder:
        if name not in self._embedders:
            self._embedders[name] = get_embedder(name, dims)
        return self._embedders[name]

    # --- stores ----------------------------------------------------------------------------------

    def store_config(self, name: str) -> StoreConfig:
        return self.cfg.store(name)

    def store(self, name: str, dims: int | None = None) -> ErasableStore:
        if name in self._stores:
            return self._stores[name]
        with self._build_lock:  # two sagas in two threads must not both construct a store
            if name in self._stores:
                return self._stores[name]
            sc = self.store_config(name)
            adapter = self.build_store(sc, dims)
            self._stores[name] = adapter
            self.lineage.register_store(name, sc.kind, self.scope)
            return adapter

    def build_store(self, sc: StoreConfig, dims: int | None = None) -> ErasableStore:
        kind = sc.kind
        emb_dims = dims
        if kind in {"chroma", "faiss", "qdrant", "pgvector"} and emb_dims is None:
            assert sc.embedding is not None
            emb_dims = self.embedder(sc.embedding).dims
        try:
            if kind == "chroma":
                from tombstone.stores.chroma import ChromaStore

                return ChromaStore(
                    sc.name,
                    _abs(self.root, sc.path),
                    collection=sc.collection or "tombstone-kb",
                    embedding_model=sc.embedding or "",
                    dims=emb_dims or 0,
                )
            if kind == "faiss":
                from tombstone.stores.faiss import FaissStore

                return FaissStore(
                    sc.name,
                    _abs(self.root, sc.path),
                    embedding_model=sc.embedding or "",
                    dims=emb_dims or 0,
                )
            if kind == "qdrant":
                from tombstone.stores.qdrant import QdrantStore

                return QdrantStore(
                    sc.name,
                    path=_abs(self.root, sc.path) if sc.path else None,
                    url=sc.url,
                    collection=sc.collection or "kb",
                    embedding_model=sc.embedding or "",
                    dims=emb_dims or 0,
                )
            if kind == "pgvector":
                from tombstone.stores.pgvector import PgVectorStore

                assert sc.dsn is not None and sc.table is not None
                return PgVectorStore(
                    sc.name,
                    sc.dsn,
                    sc.table,
                    embedding_model=sc.embedding or "",
                    dims=emb_dims or 0,
                )
            if kind == "docstore":
                from tombstone.stores.docstore import SQLiteDocStore

                return SQLiteDocStore(sc.name, _abs(self.root, sc.path))
            if kind == "cache_exact":
                from tombstone.stores.cache_exact import TombstoneExactCache

                return TombstoneExactCache(sc.name, _abs(self.root, sc.path), self.capture())
            if kind == "cache_semantic":
                from tombstone.stores.cache_semantic import SemanticCache

                assert sc.backing is not None
                backing_cfg = self.store_config(sc.backing)
                assert backing_cfg.embedding is not None
                emb = self.embedder(backing_cfg.embedding)
                backing_store = self._build_cache_backing(backing_cfg, sc.name, emb.dims)
                return SemanticCache(sc.name, backing_store, emb, self.capture())
            if kind == "dataset":
                from tombstone.train.dataset import DatasetStore

                return DatasetStore(sc.name, _abs(self.root, sc.manifest))
            if kind == "adapter":
                from tombstone.stores.adapter import AdapterStore

                ds = None
                if sc.dataset:
                    from tombstone.train.dataset import DatasetStore

                    d = self.store(sc.dataset)
                    assert isinstance(d, DatasetStore)
                    ds = d
                mia_ref = (
                    _abs(self.root, self.cfg.model.mia_reference)
                    if self.cfg.model.mia_reference
                    else None
                )
                return AdapterStore(
                    sc.name,
                    _abs(self.root, sc.path),
                    shards=sc.shards or 1,
                    base_model=sc.base_model or self.cfg.model.base,
                    runtime_model_cfg=self.cfg.model,
                    dataset=ds,
                    mia_reference=mia_ref,
                )
            if kind == "memory":
                from tombstone.stores.memory import MemoryStore

                return MemoryStore(sc.name)
        except ImportError as e:
            raise ConfigError(
                f"store {sc.name!r} of kind {kind!r} needs an optional extra that is not installed: "
                f"{e}. Install with: uv pip install 'tombstone-erase[{_extra_for(kind)}]'"
            ) from e
        raise ConfigError(f"unknown store kind {kind!r}")

    def _build_cache_backing(self, backing_cfg: StoreConfig, cache_name: str, dims: int) -> Any:
        """A semantic cache lives in its own collection/table/file on the backing backend."""
        suffix = cache_name.replace("/", "_").replace(":", "_")
        derived = backing_cfg.model_copy(
            update={
                "name": f"{backing_cfg.name}#{suffix}",
                "collection": f"{(backing_cfg.collection or 'tombstone-kb')}-{suffix}",
                "table": f"{backing_cfg.table or 'documents'}_{suffix}"
                if backing_cfg.table
                else None,
                "path": (
                    str(
                        _abs(self.root, backing_cfg.path).with_name(
                            _abs(self.root, backing_cfg.path).name + f"-{suffix}"
                        )
                    )
                    if backing_cfg.kind == "faiss" and backing_cfg.path
                    else backing_cfg.path
                ),
            }
        )
        return self.build_store(derived, dims)

    def all_stores(self) -> dict[str, ErasableStore]:
        for sc in self.cfg.stores:
            self.store(sc.name)
        return dict(self._stores)

    def close(self) -> None:
        for s in self._stores.values():
            with contextlib.suppress(Exception):
                s.close()
        self._stores.clear()
        if self._lineage is not None:
            self._lineage.close()
            self._lineage = None
        _SHARED.pop(str(self.cfg_path.resolve()), None)


def _extra_for(kind: str) -> str:
    return {
        "chroma": "chroma",
        "faiss": "faiss",
        "qdrant": "qdrant",
        "pgvector": "pgvector",
        "cache_exact": "langchain",
        "cache_semantic": "chroma",
        "adapter": "train",
    }.get(kind, "all")


@contextmanager
def runtime(explicit: str | Path | None = None) -> Iterator[Runtime]:
    rt = Runtime.load(explicit)
    try:
        yield rt
    finally:
        rt.close()
