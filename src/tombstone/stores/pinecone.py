"""Pinecone as an ``ErasableStore`` — the first backend nobody can byte-check.

Every other adapter here ends in the same place: open the files, look for the bytes, say whether
they are there. Pinecone runs on somebody else's machines. There is no persist directory, no
table to `pg_read_binary_file`, no snapshot to scan. `delete` returns success and that is the
end of the operator's access.

So this adapter reports ``{LOGICAL}`` and nothing more, and the receipt says
``UNVERIFIED-managed``. That is not the adapter giving up; it is the only true answer, and it is
worth more than the alternative. A compliance officer asking "is this person's data gone from
Pinecone?" gets, today, "the API returned success". What this gives them instead is: exactly
which vectors belonged to that person, confirmation that every retrieval path this tool can
reach now misses them, an explicit statement that the stored bytes were **not** checked because
Pinecone exposes no way to check them, and a signed record saying so. The remaining assurance is
Pinecone's own commitment, and the receipt names it as such rather than dressing it up as a
measurement.

Two behaviours of the service shape the code:

* **Deletes and metadata updates are eventually consistent.** A vector can still come back from
  a query for a short while after a successful delete. A probe run immediately would report the
  artifact as still retrievable and the lattice would call that RESIDUAL — a false alarm caused
  by propagation, not by data. Both suppress and reclaim therefore wait until the change is
  visible through the API, bounded, and record how long that took.
* **Namespaces are the unit of isolation.** One index can hold many; this adapter is bound to
  one namespace and never reads or writes another.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from tombstone.errors import NotSupported
from tombstone.lineage.capture import EmbedRecord
from tombstone.lineage.stamp import K_SUPPRESSED
from tombstone.model.artifacts import ArtifactRef
from tombstone.model.status import VerifyLevel
from tombstone.stores._vector import VectorBackendBase
from tombstone.stores.base import Hit, ReclaimResult

# Pinecone metadata keys cannot contain a dot in filter expressions the way our stamp keys do,
# and the service rejects null values outright.
_DOC_KEY = "__document"


def _enc_key(k: str) -> str:
    return k.replace(".", "__")


def _dec_key(k: str) -> str:
    return k.replace("__", ".")


def _enc_md(md: Mapping[str, Any]) -> dict[str, Any]:
    """Pinecone accepts strings, numbers, booleans and lists of strings. Anything else is
    stringified rather than dropped, so a probe on it still has something to match."""
    out: dict[str, Any] = {}
    for k, v in md.items():
        if v is None:
            continue
        if isinstance(v, bool | int | float | str):
            out[_enc_key(k)] = v
        elif isinstance(v, list | tuple) and all(isinstance(x, str) for x in v):
            out[_enc_key(k)] = list(v)
        else:
            out[_enc_key(k)] = str(v)
    return out


def _dec_md(md: Mapping[str, Any]) -> dict[str, Any]:
    return {_dec_key(k): v for k, v in md.items() if k != _DOC_KEY}


class PineconeStore(VectorBackendBase):
    """One Pinecone index namespace.

    ``settle_timeout_s`` bounds how long suppress and reclaim wait for the service to make a
    change visible. It is a wait for *consistency*, never a retry of the operation: the delete
    is issued once.
    """

    kind = "pinecone"

    def __init__(
        self,
        name: str,
        index: str,
        api_key: str = "",
        namespace: str = "",
        embedding_model: str = "",
        dims: int = 0,
        settle_timeout_s: float = 30.0,
        host: str = "",
    ) -> None:
        super().__init__(name, embedding_model, dims)
        import os

        from pinecone import Pinecone

        key = api_key or os.environ.get("PINECONE_API_KEY", "")
        if not key:
            raise NotSupported(
                "Pinecone needs an API key: set PINECONE_API_KEY, or put api_key in the store's "
                "entry in tombstone.yaml. The key is never written to lineage or a receipt."
            )
        self.index_name = index
        self.namespace = namespace
        self.settle_timeout_s = settle_timeout_s
        self._client = Pinecone(api_key=key)
        self._index = self._client.Index(host=host) if host else self._client.Index(index)
        self.capabilities = self.detect_capabilities()

    # --- capability ------------------------------------------------------------------------

    def detect_capabilities(self) -> frozenset[VerifyLevel]:
        """``{LOGICAL}``, always, and it is not a limitation of this code.

        SEMANTIC is withheld deliberately even though ``query`` works. The drift protocol is a
        measurement of what remains in the index geometry, and reporting it next to a physical
        level we could not check would invite reading it as a substitute for one. It is not.
        """
        return frozenset({VerifyLevel.LOGICAL})

    def physical_unsupported_reason(self) -> str:
        return (
            "Pinecone is a managed service: it exposes no way to read the stored bytes — no "
            "persist directory, no snapshot to scan, no file-read API — so this tool cannot "
            "check them and does not claim to. What remains is Pinecone's own deletion "
            "commitment; ask your vendor to state it in writing if you need one"
        )

    def version(self) -> str:
        return f"pinecone-client {metadata.version('pinecone')} (managed)"

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._index.close()

    # --- storage primitives ----------------------------------------------------------------

    def _add(self, records: Sequence[EmbedRecord]) -> None:
        vectors: list[dict[str, Any]] = []
        for r in records:
            md: dict[str, Any] = _enc_md(r.metadata)
            md[_DOC_KEY] = r.document or ""
            vectors.append({"id": r.key, "values": list(r.vector), "metadata": md})
        for i in range(0, len(vectors), 100):
            self._index.upsert(vectors=vectors[i : i + 100], namespace=self.namespace)

    def _get(self, keys: Sequence[str]) -> dict[str, Hit]:
        if not keys:
            return {}
        out: dict[str, Hit] = {}
        for i in range(0, len(keys), 100):
            res = self._index.fetch(ids=list(keys[i : i + 100]), namespace=self.namespace)
            for key, vec in (getattr(res, "vectors", None) or {}).items():
                md = dict(getattr(vec, "metadata", None) or {})
                out[str(key)] = Hit(str(key), 0.0, _dec_md(md), md.get(_DOC_KEY))
        return out

    def _query_raw(self, vector: Sequence[float], k: int) -> list[Hit]:
        res = self._index.query(
            vector=list(vector),
            top_k=max(1, k),
            namespace=self.namespace,
            include_metadata=True,
        )
        return [self._hit(m) for m in (getattr(res, "matches", None) or [])]

    def _filter_raw(self, key: str, value: Any, k: int) -> list[Hit]:
        # Pinecone has no "scan by metadata" call, so a filter has to ride on a vector query.
        # The query point is arbitrary — the filter does the selecting — but it must not be the
        # zero vector: cosine similarity against it is undefined and an index can reject it. A
        # unit vector is always valid, whatever the metric.
        res = self._index.query(
            vector=self._any_unit_vector(),
            top_k=max(1, k),
            namespace=self.namespace,
            filter={_enc_key(key): {"$eq": value}},
            include_metadata=True,
        )
        return [self._hit(m) for m in (getattr(res, "matches", None) or [])]

    def _any_unit_vector(self) -> list[float]:
        n = max(1, self.dims)
        return [(1.0 / n**0.5)] * n

    @staticmethod
    def _hit(match: Any) -> Hit:
        md = dict(getattr(match, "metadata", None) or {})
        return Hit(
            str(getattr(match, "id", "")),
            float(getattr(match, "score", 0.0) or 0.0),
            _dec_md(md),
            md.get(_DOC_KEY),
        )

    def _mark_suppressed(self, keys: Sequence[str]) -> None:
        for key in keys:
            with contextlib.suppress(Exception):  # a key already gone needs no flag
                self._index.update(
                    id=key,
                    set_metadata={_enc_key(K_SUPPRESSED): True},
                    namespace=self.namespace,
                )
        # The flag is what every query path filters on, so an unsettled update is a window in
        # which suppression is not yet true. Wait for it (Hard Rule 5).
        self._settle(keys, gone=False, flagged=True)

    def _native_delete(self, keys: Sequence[str]) -> str:
        if keys:
            self._index.delete(ids=list(keys), namespace=self.namespace)
        return "index.delete(ids=...) — the service reports the request accepted; the stored bytes are not observable from here"

    def _vector_of(self, key: str) -> list[float] | None:
        res = self._index.fetch(ids=[key], namespace=self.namespace)
        vec = (getattr(res, "vectors", None) or {}).get(key)
        values = getattr(vec, "values", None) if vec is not None else None
        return [float(x) for x in values] if values else None

    def all_keys(self) -> list[str]:
        keys: list[str] = []
        with contextlib.suppress(Exception):
            for page in self._index.list(namespace=self.namespace):
                keys.extend(str(k) for k in page)
        return keys

    def count(self) -> int:
        """Vectors in *this* namespace. An index can hold others; they are not ours to count,
        and a total that silently included them would misreport coverage."""
        stats = self._index.describe_index_stats()
        spaces = getattr(stats, "namespaces", None) or {}
        if spaces:
            here = spaces.get(self.namespace)
            return int(getattr(here, "vector_count", 0) or 0) if here is not None else 0
        # an index the service reports no namespace breakdown for holds only the default one
        return int(getattr(stats, "total_vector_count", 0) or 0) if not self.namespace else 0

    def persisted_files(self) -> list[Path]:
        """None, and that is the whole point of this adapter."""
        return []

    # --- eventual consistency ----------------------------------------------------------------

    def _settle(self, keys: Sequence[str], gone: bool, flagged: bool = False) -> float:
        """Wait until the service agrees the change happened. Returns seconds waited.

        Bounded: a service that never converges returns control rather than hanging, and the
        probe that follows will report what is actually true at that moment.
        """
        if not keys:
            return 0.0
        started = time.monotonic()
        deadline = started + self.settle_timeout_s
        delay = 0.2
        while time.monotonic() < deadline:
            present = self._get(keys)
            if gone and not present:
                break
            if flagged and all(bool(h.metadata.get(K_SUPPRESSED)) for h in present.values()):
                break
            time.sleep(delay)
            delay = min(delay * 1.5, 2.0)
        return time.monotonic() - started

    # --- reclaim ------------------------------------------------------------------------------

    def _reclaim(self, refs: Sequence[ArtifactRef]) -> ReclaimResult:
        keys = [r.store_key for r in refs]
        present = self._get(keys)
        to_delete = [k for k in keys if k in present]
        if not to_delete:
            return ReclaimResult(
                noop=True,
                method="index.delete(ids=...)",
                measurement={"deleted": 0.0, "settle_s": 0.0},
                detail="every id was already absent from this namespace",
            )
        self._index.delete(ids=to_delete, namespace=self.namespace)
        waited = self._settle(to_delete, gone=True)
        left = self._get(to_delete)
        return ReclaimResult(
            noop=False,
            method="index.delete(ids=...) — bytes not observable",
            measurement={
                "deleted": float(len(to_delete)),
                "settle_s": round(waited, 3),
                "still_returned": float(len(left)),
            },
            detail=(
                "the service accepted the delete and stopped returning these ids"
                if not left
                else f"{len(left)} id(s) were still returned after {waited:.1f}s; deletes here "
                "are eventually consistent, so this is reported, not retried"
            ),
        )

    def probe_physical(self, ref: ArtifactRef) -> Any:
        raise NotSupported(
            f"store {self.name!r} cannot be physically verified: {self.physical_unsupported_reason()}"
        )
