"""Chroma (embedded ``PersistentClient``) as an ``ErasableStore``.

Native ``delete()`` in Chroma is a soft delete: hnswlib marks the label deleted and the vector
bytes stay in the segment's ``data_level0.bin``; the SQLite file keeps the row bytes in free
pages and the embeddings queue until purged. Reclaim = rewrite the collection from survivors
into a fresh segment, drop the old one, purge the queue and VACUUM the SQLite file.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from tombstone.lineage.capture import EmbedRecord
from tombstone.lineage.stamp import K_SUPPRESSED
from tombstone.model.artifacts import ArtifactRef
from tombstone.model.status import VerifyLevel
from tombstone.stores._vector import VectorBackendBase
from tombstone.stores.base import Hit, ReclaimResult
from tombstone.verify.physical import walk_files


def _clean_md(md: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in md.items():
        if isinstance(v, bool | int | float | str):
            out[k] = v
        elif v is None:
            continue
        else:
            out[k] = str(v)
    return out


class ChromaStore(VectorBackendBase):
    kind = "chroma"

    def __init__(
        self,
        name: str,
        path: str | Path,
        collection: str = "tombstone-kb",
        embedding_model: str = "",
        dims: int = 0,
    ) -> None:
        super().__init__(name, embedding_model, dims)
        import chromadb
        from chromadb.config import Settings

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.path), settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection
        self._coll = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )
        self.capabilities = self.detect_capabilities()

    def detect_capabilities(self) -> frozenset[VerifyLevel]:
        caps = {VerifyLevel.LOGICAL}
        if (self.path / "chroma.sqlite3").is_file() and os.access(self.path, os.W_OK | os.R_OK):
            caps.add(VerifyLevel.PHYSICAL)
            caps.add(VerifyLevel.SEMANTIC)
        return frozenset(caps)

    def physical_unsupported_reason(self) -> str:
        return f"persist directory {self.path} is not readable/writable by this process"

    def version(self) -> str:
        import chromadb

        return f"chromadb {chromadb.__version__}"

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.clear_system_cache()

    # --- primitives ------------------------------------------------------------------------------

    def _add(self, records: Sequence[EmbedRecord]) -> None:
        for i in range(0, len(records), 500):
            batch = records[i : i + 500]
            self._coll.upsert(
                ids=[r.key for r in batch],
                embeddings=cast(Any, [r.vector for r in batch]),
                metadatas=cast(Any, [_clean_md(r.metadata) for r in batch]),
                documents=[r.document or "" for r in batch],
            )

    def _get(self, keys: Sequence[str]) -> dict[str, Hit]:
        if not keys:
            return {}
        res = self._coll.get(ids=list(keys), include=["metadatas", "documents"])
        out: dict[str, Hit] = {}
        for i, key in enumerate(res["ids"]):
            md = (res.get("metadatas") or [{}])[i] or {}
            docs = res.get("documents") or []
            out[str(key)] = Hit(str(key), 0.0, dict(md), docs[i] if i < len(docs) else None)
        return out

    def _query_raw(self, vector: Sequence[float], k: int) -> list[Hit]:
        n = self._coll.count()
        if n == 0:
            return []
        res = self._coll.query(
            query_embeddings=cast(Any, [list(vector)]),
            n_results=min(k, n),
            include=["metadatas", "documents", "distances"],
        )
        out: list[Hit] = []
        ids = res["ids"][0]
        mds = (res.get("metadatas") or [[]])[0] or []
        docs = (res.get("documents") or [[]])[0] or []
        dists = (res.get("distances") or [[]])[0] or []
        for i, key in enumerate(ids):
            out.append(
                Hit(
                    str(key),
                    float(dists[i]) if i < len(dists) else 0.0,
                    dict(mds[i] or {}) if i < len(mds) else {},
                    docs[i] if i < len(docs) else None,
                )
            )
        return out

    def _filter_raw(self, key: str, value: Any, k: int) -> list[Hit]:
        res = self._coll.get(where={key: value}, limit=k, include=["metadatas", "documents"])
        out: list[Hit] = []
        for i, kid in enumerate(res["ids"]):
            mds = res.get("metadatas") or []
            docs = res.get("documents") or []
            out.append(
                Hit(
                    str(kid),
                    0.0,
                    dict(mds[i] or {}) if i < len(mds) else {},
                    docs[i] if i < len(docs) else None,
                )
            )
        return out

    def _mark_suppressed(self, keys: Sequence[str]) -> None:
        if not keys:
            return
        existing = self._get(keys)
        ids = [k for k in keys if k in existing]
        if not ids:
            return
        mds = []
        for k in ids:
            md = dict(existing[k].metadata)
            md[K_SUPPRESSED] = True
            mds.append(_clean_md(md))
        self._coll.update(ids=ids, metadatas=cast(Any, mds))

    def _native_delete(self, keys: Sequence[str]) -> str:
        if keys:
            self._coll.delete(ids=list(keys))
        return "collection.delete(ids=...) (hnswlib mark_deleted; bytes remain)"

    def _vector_of(self, key: str) -> list[float] | None:
        res = self._coll.get(ids=[key], include=["embeddings"])
        embs = res.get("embeddings")
        if embs is None or len(embs) == 0:
            return None
        return [float(x) for x in embs[0]]

    def all_keys(self) -> list[str]:
        n = self._coll.count()
        if n == 0:
            return []
        res = self._coll.get(limit=n, include=[])
        return [str(i) for i in res["ids"]]

    def count(self) -> int:
        return int(self._coll.count())

    def persisted_files(self) -> list[Path]:
        return walk_files(self.path)

    def _sqlite_path(self) -> Path:
        return self.path / "chroma.sqlite3"

    def _reclaim(self, refs: Sequence[ArtifactRef]) -> ReclaimResult:
        keys = [r.store_key for r in refs]
        present = self._get(keys)
        to_delete = [k for k in keys if k in present]
        residue = self._residue_present(refs)
        if not to_delete and residue is False:
            return ReclaimResult(
                noop=True,
                method="compact + rewrite segment",
                measurement={
                    "deleted": 0.0,
                    "survivors": float(self.count()),
                    "wal_rows_purged": 0.0,
                },
                detail="nothing to delete and no residue found",
            )
        # 1. logical delete of what's left
        if to_delete:
            self._coll.delete(ids=to_delete)
        # 2. rewrite the collection into a fresh segment from survivors
        survivors = self._dump_all()
        meta = dict(self._coll.metadata or {})
        self._client.delete_collection(self.collection_name)
        self._coll = self._client.get_or_create_collection(
            self.collection_name,
            metadata=meta or {"hnsw:space": "cosine"},
            embedding_function=None,
        )
        for i in range(0, len(survivors["ids"]), 500):
            sl = slice(i, i + 500)
            if not survivors["ids"][sl]:
                break
            self._coll.upsert(
                ids=survivors["ids"][sl],
                embeddings=cast(Any, survivors["embeddings"][sl]),
                metadatas=cast(Any, survivors["metadatas"][sl]),
                documents=survivors["documents"][sl],
            )
        # 3. purge the embeddings queue (Chroma's write-ahead log inside sqlite) and VACUUM
        purged = self._purge_sqlite()
        # Chroma flushes segments from a background thread; an orphan segment directory of the
        # old collection can reappear for a moment after delete_collection. Settle, then sweep.
        import time

        for _ in range(5):
            self._remove_orphan_segments()
            time.sleep(0.2)
        self._remove_orphan_segments()
        return ReclaimResult(
            noop=False,
            method="compact + rewrite segment",
            measurement={
                "deleted": float(len(to_delete)),
                "survivors": float(len(survivors["ids"])),
                "wal_rows_purged": float(purged),
            },
        )

    def _dump_all(self) -> dict[str, list[Any]]:
        n = self._coll.count()
        out: dict[str, list[Any]] = {"ids": [], "embeddings": [], "metadatas": [], "documents": []}
        if n == 0:
            return out
        res = self._coll.get(limit=n, include=["embeddings", "metadatas", "documents"])
        out["ids"] = [str(i) for i in res["ids"]]
        embs = res.get("embeddings")
        out["embeddings"] = [[float(x) for x in e] for e in (embs if embs is not None else [])]
        out["metadatas"] = [dict(m or {}) for m in (res.get("metadatas") or [])]
        out["documents"] = [d or "" for d in (res.get("documents") or [])]
        return out

    def _purge_sqlite(self) -> int:
        """Delete queued (already-applied) embedding rows for collections that no longer exist,
        then VACUUM so freed pages are actually rewritten. Chroma normally purges its queue
        lazily; we do it now because the residue is the point."""
        db = self._sqlite_path()
        if not db.is_file():
            return 0
        purged = 0
        conn = sqlite3.connect(str(db), isolation_level=None, timeout=30)
        try:
            tables = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            live_segments: set[str] = set()
            if "segments" in tables:
                live_segments = {
                    str(r[0]) for r in conn.execute("SELECT id FROM segments").fetchall()
                }
            if "embeddings_queue" in tables:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(embeddings_queue)")]
                if "topic" in cols:
                    rows = conn.execute("SELECT seq_id, topic FROM embeddings_queue").fetchall()
                    dead = [
                        r[0]
                        for r in rows
                        if not any(seg in str(r[1]) for seg in live_segments)
                        and not self._topic_is_live(str(r[1]), conn)
                    ]
                    for seq in dead:
                        conn.execute("DELETE FROM embeddings_queue WHERE seq_id = ?", (seq,))
                    purged = len(dead)
            conn.execute("VACUUM")
        finally:
            conn.close()
        return purged

    @staticmethod
    def _topic_is_live(topic: str, conn: sqlite3.Connection) -> bool:
        # topic looks like persistent://default/default/<collection uuid>
        cid = topic.rsplit("/", 1)[-1]
        row = conn.execute("SELECT 1 FROM collections WHERE id = ?", (cid,)).fetchone()
        return row is not None

    def _remove_orphan_segments(self) -> None:
        """Remove segment directories no longer referenced by the sqlite catalogue."""
        db = self._sqlite_path()
        if not db.is_file():
            return
        conn = sqlite3.connect(str(db), timeout=30)
        try:
            live = {str(r[0]) for r in conn.execute("SELECT id FROM segments").fetchall()}
        except sqlite3.Error:
            return
        finally:
            conn.close()
        import shutil

        for d in self.path.iterdir():
            if d.is_dir() and d.name not in live and (d / "header.bin").exists():
                shutil.rmtree(d, ignore_errors=True)
