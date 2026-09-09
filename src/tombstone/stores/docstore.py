"""A small SQLite document store: the app's "source rows" and chunk text, keyed by artifact id.

It exists so SOURCE and CHUNK artifacts have a real store that can be suppressed, reclaimed and
probed. Physical probe = the artifact id string (present in every row) is absent from the file
bytes after ``DELETE`` + ``VACUUM``.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tombstone.model.artifacts import ArtifactRef
from tombstone.model.status import VerifyLevel
from tombstone.stores.base import (
    LogicalProbeResult,
    PhysicalProbeResult,
    ProbeSet,
    ReclaimResult,
)
from tombstone.verify.physical import scan_files_for_patterns


class SQLiteDocStore:
    kind = "docstore"

    def __init__(self, name: str, path: str | Path) -> None:
        self.name = name
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=DELETE")  # no WAL: one file to scan
        self._conn.execute("PRAGMA secure_delete=OFF")  # deliberately default: residue is real
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS docs (artifact_id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
            "text TEXT NOT NULL, metadata TEXT NOT NULL, suppressed INTEGER NOT NULL DEFAULT 0)"
        )
        self.capabilities: frozenset[VerifyLevel] = self.detect_capabilities()

    def detect_capabilities(self) -> frozenset[VerifyLevel]:
        caps = {VerifyLevel.LOGICAL}
        try:
            with self.path.open("rb"):
                pass
            if self.path.parent.exists() and __import__("os").access(self.path, 2):
                caps.add(VerifyLevel.PHYSICAL)
        except OSError:
            pass
        return frozenset(caps)

    def version(self) -> str:
        return f"sqlite {sqlite3.sqlite_version}"

    def close(self) -> None:
        self._conn.close()

    # --- app-facing --------------------------------------------------------------------------

    def put(self, artifact_id: str, kind: str, text: str, metadata: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO docs (artifact_id, kind, text, metadata, suppressed) "
            "VALUES (?, ?, ?, ?, 0)",
            (artifact_id, kind, text, json.dumps(metadata, sort_keys=True)),
        )

    def get(self, artifact_id: str, include_suppressed: bool = False) -> tuple[str, dict[str, Any]] | None:
        row = self._conn.execute(
            "SELECT text, metadata, suppressed FROM docs WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        if row is None or (row[2] and not include_suppressed):
            return None
        return str(row[0]), dict(json.loads(row[1]))

    def native_delete(self, artifact_ids: Sequence[str]) -> None:
        """The 'source row dropped' path: a plain DELETE, no VACUUM."""
        self._conn.executemany("DELETE FROM docs WHERE artifact_id = ?", [(a,) for a in artifact_ids])

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0])

    def sample_keys(self, n: int) -> list[str]:
        rows = self._conn.execute("SELECT artifact_id FROM docs ORDER BY artifact_id LIMIT ?", (n,)).fetchall()
        return [str(r[0]) for r in rows]

    # --- ErasableStore ---------------------------------------------------------------------------

    def suppress(self, refs: Sequence[ArtifactRef]) -> None:
        self._conn.executemany(
            "UPDATE docs SET suppressed = 1 WHERE artifact_id = ?", [(r.store_key,) for r in refs]
        )

    def reclaim(self, refs: Sequence[ArtifactRef]) -> ReclaimResult:
        keys = [r.store_key for r in refs]
        before = self.count()
        self._conn.executemany("DELETE FROM docs WHERE artifact_id = ?", [(k,) for k in keys])
        deleted = before - self.count()
        self._conn.execute("VACUUM")
        return ReclaimResult(
            noop=deleted == 0,
            method="DELETE + VACUUM",
            measurement={"deleted": float(deleted)},
        )

    def probe_logical(self, ref: ArtifactRef, probes: ProbeSet) -> LogicalProbeResult:
        row = self._conn.execute(
            "SELECT suppressed FROM docs WHERE artifact_id = ?", (ref.store_key,)
        ).fetchone()
        found = row is not None and not row[0]
        return LogicalProbeResult(found=found, found_by=("id",) if found else (), probes_run=1)

    def probe_physical(self, ref: ArtifactRef) -> PhysicalProbeResult:
        # Every row carries its artifact id (primary key) and metadata JSON with the same id.
        return scan_files_for_patterns(
            [self.path], {"artifact_id": ref.store_key.encode("utf-8")}, method="byte-scan"
        )
