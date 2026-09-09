"""LangChain ``SQLiteCache`` wrapped so every cached answer has lineage.

The app renders retrieved chunks with ``TombstoneVectorStore.context_block(docs)``, which appends
``<!-- tombstone:chunks=ID,ID -->`` to the prompt. On ``update()`` this wrapper parses those ids,
creates a CACHE node with one edge per parent chunk, and records the cache key in a side table
inside the cache database (the app's file, not ``.tombstone/``) so the row can be found and
deleted by key later without storing the prompt anywhere else.

Caches have no useful "hide" state: suppression deletes the row immediately. Reclaim VACUUMs
the file. Physical probe = the cache key and the chunk-id delimiter are absent from the file.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache

from tombstone.lineage.capture import Capture
from tombstone.model.artifacts import ArtifactRef
from tombstone.model.status import VerifyLevel
from tombstone.stores.base import (
    LogicalProbeResult,
    PhysicalProbeResult,
    ProbeSet,
    ReclaimResult,
)
from tombstone.util import sha256_hex
from tombstone.verify.physical import scan_files_for_patterns

_DELIM_RE = re.compile(r"<!-- tombstone:chunks=([0-9A-Z,]*) -->")


def cache_key(prompt: str, llm_string: str) -> str:
    return sha256_hex(prompt + "\x1f" + llm_string)[:40]


def parse_chunk_ids(prompt: str) -> list[str]:
    ids: list[str] = []
    for m in _DELIM_RE.finditer(prompt):
        ids.extend(i for i in m.group(1).split(",") if i)
    return sorted(set(ids))


class TombstoneExactCache(BaseCache):
    kind = "cache_exact"

    def __init__(self, name: str, path: str | Path, capture: Capture) -> None:
        from langchain_community.cache import SQLiteCache

        self.name = name
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._inner = SQLiteCache(database_path=str(self.path))
        self.capture = capture
        capture.register(name, self.kind)
        self._lock = threading.RLock()
        self._side = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self._x(
            "CREATE TABLE IF NOT EXISTS tombstone_cache_keys (key TEXT PRIMARY KEY, "
            "prompt TEXT NOT NULL, llm TEXT NOT NULL)"
        )
        self.capabilities: frozenset[VerifyLevel] = frozenset(
            {VerifyLevel.LOGICAL, VerifyLevel.PHYSICAL}
        )

    def _x(self, sql: str, params: Any = ()) -> Any:
        with self._lock:
            return self._side.execute(sql, params)

    def version(self) -> str:
        return f"langchain SQLiteCache on sqlite {sqlite3.sqlite_version}"

    def close(self) -> None:
        self._side.close()

    # --- BaseCache -----------------------------------------------------------------------------

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        with self._lock:
            return self._inner.lookup(prompt, llm_string)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        with self._lock:
            self._inner.update(prompt, llm_string, return_val)
        key = cache_key(prompt, llm_string)
        self._x(
            "INSERT OR REPLACE INTO tombstone_cache_keys (key, prompt, llm) VALUES (?, ?, ?)",
            (key, prompt, llm_string),
        )
        parents = parse_chunk_ids(prompt)
        answer = "".join(getattr(g, "text", str(g)) for g in return_val)
        self.capture.record_cache(
            self.name, key, parents, sha256_hex(answer), subject=None, via="cache:exact"
        )

    def clear(self, **kwargs: Any) -> None:
        self._inner.clear()
        self._x("DELETE FROM tombstone_cache_keys")

    # --- ErasableStore -------------------------------------------------------------------------

    @staticmethod
    def _key_of(ref: ArtifactRef) -> str:
        return ref.store_key.split("@", 1)[0]

    def _delete_keys(self, keys: Sequence[str]) -> int:
        n = 0
        for key in keys:
            row = self._x(
                "SELECT prompt, llm FROM tombstone_cache_keys WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                continue
            self._x("DELETE FROM full_llm_cache WHERE prompt = ? AND llm = ?", (row[0], row[1]))
            self._x("DELETE FROM tombstone_cache_keys WHERE key = ?", (key,))
            n += 1
        return n

    def suppress(self, refs: Sequence[ArtifactRef]) -> None:
        self._delete_keys([self._key_of(r) for r in refs])

    def reclaim(self, refs: Sequence[ArtifactRef]) -> ReclaimResult:
        deleted = self._delete_keys([self._key_of(r) for r in refs])
        self._x("VACUUM")
        return ReclaimResult(
            noop=deleted == 0, method="DELETE + VACUUM", measurement={"deleted": float(deleted)}
        )

    def probe_logical(self, ref: ArtifactRef, probes: ProbeSet) -> LogicalProbeResult:
        row = self._x(
            "SELECT prompt, llm FROM tombstone_cache_keys WHERE key = ?", (self._key_of(ref),)
        ).fetchone()
        found = row is not None and self._inner.lookup(row[0], row[1]) is not None
        return LogicalProbeResult(found=found, found_by=("id",) if found else (), probes_run=1)

    def probe_physical(self, ref: ArtifactRef) -> PhysicalProbeResult:
        return scan_files_for_patterns(
            [self.path], {"cache_key": self._key_of(ref).encode("utf-8")}, method="byte-scan"
        )

    def count(self) -> int:
        return int(self._x("SELECT COUNT(*) FROM tombstone_cache_keys").fetchone()[0])

    def sample_keys(self, n: int) -> list[str]:
        rows = self._x("SELECT key FROM tombstone_cache_keys ORDER BY key LIMIT ?", (n,)).fetchall()
        return [str(r[0]) for r in rows]
