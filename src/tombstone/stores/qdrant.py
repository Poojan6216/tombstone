"""Qdrant as an ``ErasableStore`` — local mode (``path=``) for tests/bench, ``url=`` for servers.

Local mode persists points as pickled Python objects in ``storage.sqlite`` files: a deleted point
is removed from its row, but SQLite keeps the page bytes until VACUUM, and the pickle encodes
floats as big-endian float64 (derived exactly from the float32 values we stored). Reclaim
(local) = delete + rewrite the collection from survivors + VACUUM. Reclaim (server) = delete +
optimizer trigger, then a snapshot scan when the snapshot API is reachable.

Payload keys containing dots are nested paths in Qdrant filters, so ``tombstone.x`` is stored
as ``tombstone__x`` and translated back on read.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import uuid
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, cast

from tombstone.lineage.capture import EmbedRecord
from tombstone.lineage.stamp import K_SUPPRESSED
from tombstone.model.status import VerifyLevel
from tombstone.stores._vector import VectorBackendBase
from tombstone.stores.base import Hit, ReclaimResult
from tombstone.util import fingerprint_bytes
from tombstone.verify.physical import fingerprint_patterns, walk_files

_NS = uuid.UUID("6d1a2c0e-9d0f-4d6b-8c3a-1f2e3d4c5b6a")
_DOC_KEY = "__document"
_KEY_KEY = "__key"


def _enc(md: dict[str, Any]) -> dict[str, Any]:
    return {k.replace(".", "__"): v for k, v in md.items()}


def _dec(payload: dict[str, Any]) -> dict[str, Any]:
    return {k.replace("__", "."): v for k, v in payload.items() if not k.startswith("__")}


def _point_id(key: str) -> str:
    return str(uuid.uuid5(_NS, key))


class QdrantStore(VectorBackendBase):
    kind = "qdrant"

    def __init__(
        self,
        name: str,
        path: str | Path | None = None,
        url: str | None = None,
        collection: str = "kb",
        embedding_model: str = "",
        dims: int = 0,
        api_key: str | None = None,
    ) -> None:
        super().__init__(name, embedding_model, dims)
        from qdrant_client import QdrantClient, models

        self._models = models
        self.collection = collection
        self.path = Path(path) if path else None
        self.url = url
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.path))
        elif url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            raise ValueError("QdrantStore needs path= (local) or url= (server)")
        if not self._client.collection_exists(collection):
            if dims <= 0:
                raise ValueError("dims is required to create a new Qdrant collection")
            self._client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=dims, distance=models.Distance.COSINE),
            )
        else:
            info = self._client.get_collection(collection)
            params = info.config.params.vectors
            size = getattr(params, "size", None)
            if isinstance(size, int):
                self.dims = size
        self.capabilities = self.detect_capabilities()

    def detect_capabilities(self) -> frozenset[VerifyLevel]:
        caps = {VerifyLevel.LOGICAL}
        if self.path is not None:
            if os.access(self.path, os.R_OK | os.W_OK):
                caps.add(VerifyLevel.PHYSICAL)
                caps.add(VerifyLevel.SEMANTIC)
        else:
            with contextlib.suppress(Exception):
                self._client.list_snapshots(self.collection)
                caps.add(VerifyLevel.PHYSICAL)
                caps.add(VerifyLevel.SEMANTIC)
        return frozenset(caps)

    def physical_unsupported_reason(self) -> str:
        if self.path is not None:
            return f"local storage {self.path} is not readable/writable"
        return "snapshot API not reachable on this server (needs snapshot permissions)"

    def version(self) -> str:
        return f"qdrant-client {metadata.version('qdrant-client')} ({'local' if self.path else 'server'})"

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.close()

    # --- primitives ------------------------------------------------------------------------------

    def _add(self, records: Sequence[EmbedRecord]) -> None:
        pts = []
        for r in records:
            payload = _enc(r.metadata)
            payload[_KEY_KEY] = r.key
            payload[_DOC_KEY] = r.document or ""
            pts.append(
                self._models.PointStruct(
                    id=_point_id(r.key), vector=list(r.vector), payload=payload
                )
            )
        for i in range(0, len(pts), 256):
            self._client.upsert(self.collection, points=pts[i : i + 256], wait=True)

    def _to_hit(self, point: Any, score: float = 0.0) -> Hit:
        payload = dict(point.payload or {})
        key = str(payload.get(_KEY_KEY, point.id))
        return Hit(key, score, _dec(payload), payload.get(_DOC_KEY))

    def _get(self, keys: Sequence[str]) -> dict[str, Hit]:
        if not keys:
            return {}
        pts = self._client.retrieve(
            self.collection, ids=[_point_id(k) for k in keys], with_payload=True, with_vectors=False
        )
        return {(h := self._to_hit(p)).key: h for p in pts}

    def _query_raw(self, vector: Sequence[float], k: int) -> list[Hit]:
        res = self._client.query_points(
            self.collection, query=list(vector), limit=k, with_payload=True
        )
        return [self._to_hit(p, float(p.score)) for p in res.points]

    def _filter_raw(self, key: str, value: Any, k: int) -> list[Hit]:
        flt = self._models.Filter(
            must=[
                self._models.FieldCondition(
                    key=key.replace(".", "__"), match=self._models.MatchValue(value=value)
                )
            ]
        )
        pts, _ = self._client.scroll(
            self.collection, scroll_filter=flt, limit=k, with_payload=True, with_vectors=False
        )
        return [self._to_hit(p) for p in pts]

    def _mark_suppressed(self, keys: Sequence[str]) -> None:
        if keys:
            self._client.set_payload(
                self.collection,
                payload={K_SUPPRESSED.replace(".", "__"): True},
                points=[_point_id(k) for k in keys],
                wait=True,
            )

    def _native_delete(self, keys: Sequence[str]) -> str:
        if keys:
            self._client.delete(
                self.collection,
                points_selector=self._models.PointIdsList(points=[_point_id(k) for k in keys]),
                wait=True,
            )
        return "client.delete(points) (row removed; page bytes remain until VACUUM/optimizer)"

    def _vector_of(self, key: str) -> list[float] | None:
        pts = self._client.retrieve(
            self.collection, ids=[_point_id(key)], with_payload=False, with_vectors=True
        )
        if not pts:
            return None
        v = pts[0].vector
        if isinstance(v, list) and v and not isinstance(v[0], list):
            return [float(cast(Any, x)) for x in v]
        return None

    def all_keys(self) -> list[str]:
        keys: list[str] = []
        offset = None
        while True:
            pts, offset = self._client.scroll(
                self.collection,
                limit=1000,
                offset=offset,
                with_payload=[_KEY_KEY],
                with_vectors=False,
            )
            keys.extend(str((p.payload or {}).get(_KEY_KEY, p.id)) for p in pts)
            if offset is None:
                break
        return keys

    def count(self) -> int:
        return int(self._client.count(self.collection, exact=True).count)

    def persisted_files(self) -> list[Path]:
        if self.path is not None:
            return walk_files(self.path)
        return []

    def _encoding_patterns(self, fingerprint_hex: str) -> dict[str, bytes]:
        pats = fingerprint_patterns(fingerprint_hex)
        # local mode pickles python floats (BINFLOAT opcodes); a server stores float32.
        return {
            "f32le": fingerprint_bytes(fingerprint_hex),
            "pickle_binfloat": pats["pickle_binfloat"],
        }

    def _reclaim(self, keys: Sequence[str]) -> ReclaimResult:
        present = self._get(keys)
        to_delete = [k for k in keys if k in present]
        if to_delete:
            self._native_delete(to_delete)
        if self.path is not None:
            # Rewrite the collection from survivors so no stale page remains, then VACUUM.
            survivors = self._dump_all()
            self._client.delete_collection(self.collection)
            self._client.create_collection(
                self.collection,
                vectors_config=self._models.VectorParams(
                    size=self.dims, distance=self._models.Distance.COSINE
                ),
            )
            for i in range(0, len(survivors), 256):
                self._client.upsert(self.collection, points=survivors[i : i + 256], wait=True)
            vacuumed = self._vacuum_local()
            return ReclaimResult(
                noop=not to_delete,
                method="delete + rewrite collection from survivors + VACUUM storage.sqlite",
                measurement={
                    "deleted": float(len(to_delete)),
                    "survivors": float(len(survivors)),
                    "sqlite_files_vacuumed": float(vacuumed),
                },
            )
        # Server: ask the optimizer to rewrite segments, then wait for green.
        self._client.update_collection(
            self.collection,
            optimizers_config=self._models.OptimizersConfigDiff(
                deleted_threshold=0.0, vacuum_min_vector_number=1
            ),
        )
        import time

        for _ in range(60):
            info = self._client.get_collection(self.collection)
            if str(info.status).lower().endswith("green"):
                break
            time.sleep(0.5)
        return ReclaimResult(
            noop=not to_delete,
            method="delete + optimizer (deleted_threshold=0) + wait green",
            measurement={"deleted": float(len(to_delete))},
        )

    def _dump_all(self) -> list[Any]:
        pts_all: list[Any] = []
        offset = None
        while True:
            pts, offset = self._client.scroll(
                self.collection, limit=500, offset=offset, with_payload=True, with_vectors=True
            )
            for p in pts:
                pts_all.append(
                    self._models.PointStruct(id=p.id, vector=p.vector, payload=p.payload or {})
                )
            if offset is None:
                break
        return pts_all

    def _vacuum_local(self) -> int:
        n = 0
        assert self.path is not None
        # The local client keeps sqlite connections open; close, VACUUM every sqlite, reopen.
        self._client.close()
        for f in walk_files(self.path):
            if f.suffix == ".sqlite":
                conn = sqlite3.connect(str(f), isolation_level=None)
                try:
                    conn.execute("VACUUM")
                    n += 1
                finally:
                    conn.close()
        from qdrant_client import QdrantClient

        self._client = QdrantClient(path=str(self.path))
        return n
