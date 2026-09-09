"""FAISS ``IndexHNSWFlat`` wrapped in ``IndexIDMap2``, persisted to a file, with a JSON sidecar
for metadata/documents and a persisted exclusion set for suppression.

HNSW cannot remove vectors: ``remove_ids`` raises, so the "native delete" every app performs is
to drop the id mapping (LangChain does exactly this in its docstore) and leave the bytes in
the index file. Reclaim = rebuild the index from survivors and atomically replace the file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tombstone.lineage.capture import EmbedRecord
from tombstone.lineage.stamp import K_SUPPRESSED
from tombstone.model.artifacts import ArtifactRef
from tombstone.model.status import VerifyLevel
from tombstone.stores._vector import VectorBackendBase
from tombstone.stores.base import Hit, ReclaimResult


class FaissStore(VectorBackendBase):
    kind = "faiss"

    def __init__(
        self, name: str, path: str | Path, embedding_model: str = "", dims: int = 0
    ) -> None:
        super().__init__(name, embedding_model, dims)
        import faiss
        import numpy as np

        self._faiss = faiss
        self._np = np
        # faiss-cpu ships its own libomp; alongside torch/onnxruntime on macOS the two OpenMP
        # runtimes crash the process on the first parallel search. Pin FAISS to one thread
        # unless the operator opts in (TOMBSTONE_FAISS_THREADS). Indexes here are small.
        faiss.omp_set_num_threads(int(os.environ.get("TOMBSTONE_FAISS_THREADS", "1")))
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.path.with_suffix(self.path.suffix + ".meta.json")
        self.tomb_path = self.path.with_suffix(self.path.suffix + ".tombstones.json")
        self._meta: dict[str, dict[str, Any]] = {}
        self._next_id = 1
        self._excluded: set[int] = set()
        if self.path.is_file():
            self._index = faiss.read_index(str(self.path))
            if self.meta_path.is_file():
                blob = json.loads(self.meta_path.read_text(encoding="utf-8"))
                self._meta = blob.get("records", {})
                self._next_id = int(blob.get("next_id", 1))
            if self.tomb_path.is_file():
                self._excluded = set(json.loads(self.tomb_path.read_text(encoding="utf-8")))
            self.dims = int(self._index.d)
        else:
            if dims <= 0:
                raise ValueError("dims is required to create a new FAISS index")
            self._index = self._new_index(dims)
            self._persist()
        self.capabilities = self.detect_capabilities()

    def _new_index(self, dims: int) -> Any:
        hnsw = self._faiss.IndexHNSWFlat(dims, 32, self._faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efSearch = 128
        return self._faiss.IndexIDMap2(hnsw)

    def detect_capabilities(self) -> frozenset[VerifyLevel]:
        caps = {VerifyLevel.LOGICAL}
        if self.path.is_file() and os.access(self.path, os.R_OK | os.W_OK):
            caps.add(VerifyLevel.PHYSICAL)
            caps.add(VerifyLevel.SEMANTIC)
        return frozenset(caps)

    def physical_unsupported_reason(self) -> str:
        return f"index file {self.path} is not readable/writable by this process"

    def version(self) -> str:
        return f"faiss {getattr(self._faiss, '__version__', '?')}"

    def close(self) -> None:
        self._persist()

    # --- persistence ---------------------------------------------------------------------------

    def _persist(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self._faiss.write_index(self._index, str(tmp))
        os.replace(tmp, self.path)
        self.meta_path.write_text(
            json.dumps({"next_id": self._next_id, "records": self._meta}, sort_keys=True),
            encoding="utf-8",
        )
        self.tomb_path.write_text(json.dumps(sorted(self._excluded)), encoding="utf-8")

    def persisted_files(self) -> list[Path]:
        return [p for p in (self.path, self.meta_path, self.tomb_path) if p.is_file()]

    # --- primitives ------------------------------------------------------------------------------

    def _add(self, records: Sequence[EmbedRecord]) -> None:
        if not records:
            return
        ids = []
        for r in records:
            if r.key in self._meta:  # upsert: drop the mapping, keep the bytes (like everyone)
                self._excluded.add(int(self._meta[r.key]["id"]))
            fid = self._next_id
            self._next_id += 1
            ids.append(fid)
            self._meta[r.key] = {"id": fid, "metadata": dict(r.metadata), "document": r.document}
        x = self._np.asarray([r.vector for r in records], dtype="float32")
        self._index.add_with_ids(x, self._np.asarray(ids, dtype="int64"))
        self._persist()

    def _hit(self, key: str, score: float) -> Hit:
        rec = self._meta[key]
        return Hit(key, score, dict(rec["metadata"]), rec.get("document"))

    def _get(self, keys: Sequence[str]) -> dict[str, Hit]:
        return {k: self._hit(k, 0.0) for k in keys if k in self._meta}

    def _id_to_key(self) -> dict[int, str]:
        return {int(v["id"]): k for k, v in self._meta.items()}

    def _query_raw(self, vector: Sequence[float], k: int) -> list[Hit]:
        if self._index.ntotal == 0:
            return []
        x = self._np.asarray([list(vector)], dtype="float32")
        kk = min(k, int(self._index.ntotal))
        scores, ids = self._index.search(x, kk)
        id2key = self._id_to_key()
        out: list[Hit] = []
        for s, i in zip(scores[0], ids[0], strict=True):
            if i < 0 or int(i) not in id2key or int(i) in self._excluded:
                continue
            out.append(self._hit(id2key[int(i)], float(s)))
        return out

    def _filter_raw(self, key: str, value: Any, k: int) -> list[Hit]:
        out = [
            self._hit(kk, 0.0) for kk, v in self._meta.items() if v["metadata"].get(key) == value
        ]
        return out[:k]

    def _mark_suppressed(self, keys: Sequence[str]) -> None:
        for k in keys:
            if k in self._meta:
                self._meta[k]["metadata"][K_SUPPRESSED] = True
                self._excluded.add(int(self._meta[k]["id"]))
        self._persist()

    def _native_delete(self, keys: Sequence[str]) -> str:
        ids = [int(self._meta[k]["id"]) for k in keys if k in self._meta]
        removed = "remove_ids not supported by HNSW; id mapping dropped, bytes remain"
        if ids:
            try:
                sel = self._faiss.IDSelectorBatch(self._np.asarray(ids, dtype="int64"))
                self._index.remove_ids(sel)
                removed = "remove_ids ok"
            except RuntimeError:
                pass
            for k in keys:
                self._meta.pop(k, None)
            self._excluded.update(ids)
            self._persist()
        return removed

    def _vector_of(self, key: str) -> list[float] | None:
        rec = self._meta.get(key)
        if rec is None:
            return None
        try:
            v = self._index.reconstruct(int(rec["id"]))
        except RuntimeError:
            return None
        return [float(x) for x in v]

    def all_keys(self) -> list[str]:
        return list(self._meta)

    def _reclaim(self, refs: Sequence[ArtifactRef]) -> ReclaimResult:
        keys = [r.store_key for r in refs]
        to_delete = [k for k in keys if k in self._meta]
        residue = self._residue_present(refs)
        if not to_delete and residue is False:
            return ReclaimResult(
                noop=True,
                method="rebuild IndexHNSWFlat from survivors + atomic file replace",
                measurement={"deleted": 0.0, "survivors": float(len(self._meta))},
                detail="nothing to delete and no residue found",
            )
        for k in to_delete:
            self._excluded.add(int(self._meta[k]["id"]))
            del self._meta[k]
        # Rebuild from survivors: only their vectors are copied into a fresh index file.
        survivors = list(self._meta.items())
        new_index = self._new_index(self.dims)
        if survivors:
            vecs = []
            ids = []
            for _key, rec in survivors:
                v = self._index.reconstruct(int(rec["id"]))
                vecs.append(v)
                ids.append(int(rec["id"]))
            new_index.add_with_ids(
                self._np.asarray(vecs, dtype="float32"), self._np.asarray(ids, dtype="int64")
            )
        self._index = new_index
        self._excluded = set()  # nothing excluded remains in the new file
        self._persist()
        return ReclaimResult(
            noop=False,
            method="rebuild IndexHNSWFlat from survivors + atomic file replace",
            measurement={"deleted": float(len(to_delete)), "survivors": float(len(survivors))},
        )
