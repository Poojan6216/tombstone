"""Chroma (embedded ``PersistentClient``) as an ``ErasableStore``.

Native ``delete()`` in Chroma is a soft delete: hnswlib marks the label deleted and the vector
bytes stay in the segment's ``data_level0.bin``; the SQLite file keeps the row bytes in free
pages and the embeddings queue until purged. Reclaim = rewrite the collection from survivors
into a fresh segment, drop the old one, purge the queue and VACUUM the SQLite file.

One more thing the rewrite has to clean up, found by the physical probe on Linux CI: Chroma's
local HNSW segment persists its whole allocated capacity, not just its elements. chroma-hnswlib
``malloc``s ``data_level0_memory_`` without clearing it and ``initPersistentIndex`` writes all of
``max_elements_ * size_data_per_element_`` to ``data_level0.bin``, so the slots beyond
``cur_element_count`` hold whatever the allocator handed over — on glibc, the buffer the deleted
collection's index just freed, erased vectors included. That init runs on every open until the
index reaches ``sync_threshold`` (1000 elements) and is persisted for real, so every process that
touches the collection writes its own heap into the file. The adapter zeroes the unused slots
after the rewrite, when a new segment directory appears after a write, and at open right after
forcing that init; the reclaim measurement records the byte count.
"""

from __future__ import annotations

import contextlib
import math
import os
import sqlite3
import struct
from collections.abc import Mapping, Sequence
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


# chroma-hnswlib HEADER_FIELDS, written back to back with no padding (hnswalg.h): the persistence
# version (int), then offsetLevel0_, max_elements_, cur_element_count, size_data_per_element_,
# label_offset_, offsetData_ (size_t), maxlevel_ (int), enterpoint_node_ (unsigned), maxM_, maxM0_,
# M_ (size_t), mult_ (double), ef_construction_ (size_t). 100 bytes.
_HNSW_HEADER = struct.Struct("<IQQQQQQiIQQQdQ")
_HNSW_PERSISTENCE_VERSION = 1


def _hnsw_header(seg: Path) -> dict[str, int] | None:
    """Decode a persisted segment's ``header.bin``; None when it is not the layout we know."""
    try:
        raw = (seg / "header.bin").read_bytes()
    except OSError:
        return None
    if len(raw) != _HNSW_HEADER.size:
        return None
    v = _HNSW_HEADER.unpack(raw)
    if v[0] != _HNSW_PERSISTENCE_VERSION:
        return None
    return {
        "offset_level0": v[1],
        "max_elements": v[2],
        "count": v[3],
        "per_element": v[4],
        "label_offset": v[5],
        "offset_data": v[6],
        "max_m0": v[10],
    }


def _content_matches(measurement: Mapping[str, float]) -> float:
    """Vector-pattern matches only: the artifact-id pattern is counted separately."""
    return sum(
        float(v)
        for k, v in measurement.items()
        if k.startswith("matches_") and k not in {"matches_artifact_id", "matches_id"}
    )


class ChromaStore(VectorBackendBase):
    kind = "chroma"

    def __init__(
        self,
        name: str,
        path: str | Path,
        collection: str = "tombstone-kb",
        embedding_model: str = "",
        dims: int = 0,
        read_only: bool = False,
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
        # Chroma builds the segment writer on first use; below sync_threshold that is
        # initPersistentIndex, which writes this process's uninitialised heap into the segment
        # files (see the module docstring). Take that first use now, then zero what it wrote
        # beyond the live elements, so an open never leaves memory contents on disk.
        self._scrubbed_segments: set[str] = set()
        self.read_only = read_only
        # `tombstone scan` looks at a store it was not asked to change, so it must not take even
        # this repair: zeroing the dead slots is harmless and privacy-improving, but it is still
        # a write to somebody's production files, and a command that says it only reads has to
        # mean it. The scan reports what it found instead.
        if VerifyLevel.PHYSICAL in self.capabilities and not read_only:
            with contextlib.suppress(Exception):
                self._coll.count()
            self._scrub_unused_slots()

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
        # the first write into a new collection creates its segment directory, and that is the
        # other moment Chroma writes an uninitialised buffer to disk; once per directory
        if VerifyLevel.PHYSICAL in self.capabilities:
            self._scrub_unused_slots(only_new=True)

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
                    "unused_slot_bytes_zeroed": 0.0,
                    "segments_not_scrubbed": 0.0,
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
        # 3. the fresh segment's files were just written from an uninitialised buffer: zero the
        # slots no element occupies (module docstring), before anything measures them
        scrubbed, skipped = self._scrub_unused_slots()
        # 4. purge the embeddings queue (Chroma's write-ahead log inside sqlite) and VACUUM
        purged = self._purge_sqlite()
        # Chroma flushes segments from a background thread, so an orphan segment of the old
        # collection can outlive delete_collection for a moment. This used to sleep a fixed second
        # and hope: long enough on a quiet laptop, not on a loaded CI runner, where the sweep ran
        # before the flush and left the bytes on disk. Sweep until the byte count stops falling
        # instead of guessing how long that takes — fast when it settles immediately, patient when
        # it does not, and bounded so a store that never settles still returns.
        import time

        # Stop at zero, or when the count has held still for several consecutive sweeps — a
        # background flush can pause, so "stopped falling once" is not "finished", which is how an
        # earlier attempt at this exited early and left one artifact's bytes on disk.
        deadline = time.monotonic() + 20.0
        previous: float | None = None
        unchanged = 0
        while True:
            self._remove_orphan_segments()
            more, more_skipped = self._scrub_unused_slots(only_new=True)  # one that appeared late
            scrubbed += more
            skipped += [x for x in more_skipped if x not in skipped]
            current = math.fsum(
                _content_matches(self.probe_physical(r).measurement)
                for r in refs
                if r.embedding_fingerprint
            )
            if current == 0.0 or time.monotonic() >= deadline:
                break
            unchanged = unchanged + 1 if current == previous else 0
            if unchanged >= 4:  # held still through four sweeps: as settled as it will get
                break
            previous = current
            time.sleep(0.25)
        return ReclaimResult(
            noop=False,
            method="compact + rewrite segment",
            measurement={
                "deleted": float(len(to_delete)),
                "survivors": float(len(survivors["ids"])),
                "wal_rows_purged": float(purged),
                "unused_slot_bytes_zeroed": float(scrubbed),
                "segments_not_scrubbed": float(len(skipped)),
            },
            detail="; ".join(skipped),
        )

    def _scrub_unused_slots(self, only_new: bool = False) -> tuple[int, list[str]]:
        """Zero every persisted HNSW slot past ``cur_element_count`` in each segment directory.

        Chroma never reads those slots as elements (hnswlib addresses elements below the count,
        and clears a slot before it fills it), so zeroing them changes nothing the index can see;
        it only stops the heap contents ``initPersistentIndex`` wrote there from sitting on disk.
        Anything whose layout does not match the header exactly is left alone and reported: a
        scrub that guesses could damage a live index, and the byte scan will tell the truth about
        what remains. A stretch that is already zero is read, not rewritten, so a clean index
        costs one pass over its unused region. ``only_new`` limits the pass to directories this
        store has not scrubbed before. Returns (bytes zeroed, skipped segments with reasons)."""
        zeroed = 0
        skipped: list[str] = []
        for seg in sorted(self.path.iterdir()):
            if not seg.is_dir() or not (seg / "header.bin").is_file():
                continue
            if only_new and seg.name in self._scrubbed_segments:
                continue
            h = _hnsw_header(seg)
            if h is None:
                skipped.append(f"{seg.name}: unrecognised header")
                continue
            dims_from_layout = (h["label_offset"] - h["offset_data"]) // 4
            consistent = (
                h["offset_data"] == 4 + 4 * h["max_m0"]
                and h["per_element"] == h["label_offset"] + 8
                and (h["label_offset"] - h["offset_data"]) % 4 == 0
                and (not self.dims or dims_from_layout == self.dims)
                and h["count"] <= h["max_elements"]
            )
            if not consistent:
                skipped.append(f"{seg.name}: header inconsistent with dims={self.dims}")
                continue
            seg_ok = True
            for fname, per in (("data_level0.bin", h["per_element"]), ("length.bin", 4)):
                f = seg / fname
                if not f.is_file():
                    continue
                size = f.stat().st_size
                start = (
                    h["offset_level0"] + h["count"] * per
                    if fname == "data_level0.bin"
                    else h["count"] * per
                )
                if size > h["max_elements"] * per + h["offset_level0"] or start > size:
                    skipped.append(f"{seg.name}/{fname}: size {size} does not fit the header")
                    seg_ok = False
                    continue
                if start == size:
                    continue
                with f.open("r+b") as fh:
                    pos = start
                    dirty = False
                    while pos < size:
                        fh.seek(pos)
                        block = fh.read(min(size - pos, 1 << 20))
                        if not block:
                            break
                        if any(block):
                            fh.seek(pos)
                            fh.write(b"\0" * len(block))
                            zeroed += len(block)
                            dirty = True
                        pos += len(block)
                    if dirty:
                        fh.flush()
                        os.fsync(fh.fileno())
            if seg_ok:
                self._scrubbed_segments.add(seg.name)
        return zeroed, skipped

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
