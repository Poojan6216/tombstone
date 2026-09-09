"""pgvector as an ``ErasableStore``.

Table: ``(id TEXT PK, embedding vector(d), document TEXT, metadata JSONB, tombstoned BOOL)`` with
an HNSW index. Native ``DELETE`` leaves dead tuples in the heap and the index until VACUUM /
REINDEX. Reclaim = ``DELETE`` + ``REINDEX INDEX`` + ``VACUUM FULL`` (or ``VACUUM`` + ``REINDEX``
when FULL is denied — recorded). This is the recipe ``vector-forget`` uses for pgvector; credit
to that project.

Capabilities are detected at connect: a role that cannot read relation files
(``pg_read_binary_file``) cannot be physically verified and reports ``{LOGICAL}`` — the
"managed instance" case. ``pgstattuple`` adds a dead-tuple count when installed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tombstone.config import resolve_env
from tombstone.errors import NotSupported
from tombstone.lineage.capture import EmbedRecord
from tombstone.lineage.stamp import K_SUPPRESSED
from tombstone.model.artifacts import ArtifactRef
from tombstone.model.status import VerifyLevel
from tombstone.stores._vector import VectorBackendBase
from tombstone.stores.base import Hit, PhysicalProbeResult, ReclaimResult
from tombstone.util import fingerprint_bytes
from tombstone.verify.physical import scan_file


class PgVectorStore(VectorBackendBase):
    kind = "pgvector"

    def __init__(self, name: str, dsn: str, table: str, embedding_model: str = "", dims: int = 0) -> None:
        super().__init__(name, embedding_model, dims)
        import psycopg
        from pgvector.psycopg import register_vector

        self.table = table
        self.index_name = f"{table}_embedding_hnsw"
        self.dsn = resolve_env(dsn) or dsn
        self._conn = psycopg.connect(self.dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector") if self._can_create_ext() else None
        register_vector(self._conn)
        self._ensure_table()
        self.pgstattuple = self._has_pgstattuple()
        self.maintenance = self._maintenance_rights()
        self.can_read_files = self._can_read_files()
        self.is_superuser = self._is_superuser()
        self.capabilities = self.detect_capabilities()

    # --- setup ---------------------------------------------------------------------------------

    def _can_create_ext(self) -> bool:
        try:
            row = self._conn.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if row:
                return False
            return bool(self._conn.execute("SELECT rolsuper OR rolcreatedb FROM pg_roles WHERE rolname = current_user").fetchone()[0])
        except Exception:  # noqa: BLE001
            return False

    def _ensure_table(self) -> None:
        exists = self._conn.execute("SELECT to_regclass(%s)", (self.table,)).fetchone()[0]
        if exists is None:
            if self.dims <= 0:
                raise ValueError("dims is required to create a new pgvector table")
            self._conn.execute(
                f'CREATE TABLE "{self.table}" (id TEXT PRIMARY KEY, embedding vector({self.dims}) NOT NULL, '
                "document TEXT, metadata JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "tombstoned BOOLEAN NOT NULL DEFAULT FALSE)"
            )
            self._conn.execute(
                f'CREATE INDEX "{self.index_name}" ON "{self.table}" USING hnsw (embedding vector_cosine_ops)'
            )
            self._conn.execute(
                f'CREATE INDEX "{self.table}_metadata_gin" ON "{self.table}" USING gin (metadata)'
            )
        else:
            row = self._conn.execute(
                "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'embedding'",
                (self.table,),
            ).fetchone()
            if row and int(row[0]) > 0:
                self.dims = int(row[0])

    def _has_pgstattuple(self) -> bool:
        row = self._conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'pgstattuple'").fetchone()
        if row:
            try:
                self._conn.execute(f"SELECT dead_tuple_count FROM pgstattuple(%s)", (self.table,))
                return True
            except Exception:  # noqa: BLE001
                return False
        return False

    def _maintenance_rights(self) -> str:
        row = self._conn.execute(
            "SELECT pg_get_userbyid(relowner) = current_user OR "
            "(SELECT rolsuper FROM pg_roles WHERE rolname = current_user) "
            "FROM pg_class WHERE oid = %s::regclass",
            (self.table,),
        ).fetchone()
        return "owner" if row and row[0] else "none"

    def _can_read_files(self) -> bool:
        try:
            self._conn.execute(
                "SELECT length(pg_read_binary_file(pg_relation_filepath(%s::regclass), 0, 1))",
                (self.table,),
            )
            return True
        except Exception:  # noqa: BLE001
            self._conn.rollback() if not self._conn.autocommit else None
            return False

    def _is_superuser(self) -> bool:
        row = self._conn.execute("SELECT rolsuper FROM pg_roles WHERE rolname = current_user").fetchone()
        return bool(row and row[0])

    def detect_capabilities(self) -> frozenset[VerifyLevel]:
        caps = {VerifyLevel.LOGICAL, VerifyLevel.SEMANTIC}
        if self.can_read_files:
            caps.add(VerifyLevel.PHYSICAL)
        return frozenset(caps)

    def physical_unsupported_reason(self) -> str:
        bits = []
        if not self.can_read_files:
            bits.append(
                "this role cannot read relation files (pg_read_binary_file); grant "
                "pg_read_server_files or run the check from a superuser role"
            )
        if self.maintenance != "owner":
            bits.append("this role is not the table owner, so VACUUM/REINDEX are not permitted")
        if not self.pgstattuple:
            bits.append("pgstattuple is not installed (CREATE EXTENSION pgstattuple)")
        return "; ".join(bits) or "unknown"

    def version(self) -> str:
        v = self._conn.execute("SHOW server_version").fetchone()[0]
        ext = self._conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        return f"postgres {v}, pgvector {ext[0] if ext else '?'}"

    def close(self) -> None:
        self._conn.close()

    # --- primitives ------------------------------------------------------------------------------

    def _add(self, records: Sequence[EmbedRecord]) -> None:
        import numpy as np

        with self._conn.cursor() as cur:
            cur.executemany(
                f'INSERT INTO "{self.table}" (id, embedding, document, metadata, tombstoned) '
                "VALUES (%s, %s, %s, %s::jsonb, FALSE) ON CONFLICT (id) DO UPDATE SET "
                "embedding = EXCLUDED.embedding, document = EXCLUDED.document, "
                "metadata = EXCLUDED.metadata, tombstoned = FALSE",
                [
                    (
                        r.key,
                        np.asarray(r.vector, dtype=np.float32),
                        r.document or "",
                        json.dumps(r.metadata, sort_keys=True),
                    )
                    for r in records
                ],
            )

    def _row_hit(self, row: Sequence[Any], score: float = 0.0) -> Hit:
        md = dict(row[2]) if row[2] else {}
        md[K_SUPPRESSED] = bool(row[3]) or bool(md.get(K_SUPPRESSED, False))
        return Hit(str(row[0]), score, md, row[1])

    def _get(self, keys: Sequence[str]) -> dict[str, Hit]:
        if not keys:
            return {}
        rows = self._conn.execute(
            f'SELECT id, document, metadata, tombstoned FROM "{self.table}" WHERE id = ANY(%s)',
            (list(keys),),
        ).fetchall()
        return {str(r[0]): self._row_hit(r) for r in rows}

    def _query_raw(self, vector: Sequence[float], k: int) -> list[Hit]:
        import numpy as np

        rows = self._conn.execute(
            f'SELECT id, document, metadata, tombstoned, embedding <=> %s AS d FROM "{self.table}" '
            "ORDER BY d LIMIT %s",
            (np.asarray(vector, dtype=np.float32), k),
        ).fetchall()
        return [self._row_hit(r, float(r[4])) for r in rows]

    def query(self, vector: Sequence[float], k: int, include_suppressed: bool = False) -> list[Hit]:
        """Suppression is applied in SQL (WHERE NOT tombstoned) and post-filtered."""
        if include_suppressed:
            return self._query_raw(vector, k)
        import numpy as np

        rows = self._conn.execute(
            f'SELECT id, document, metadata, tombstoned, embedding <=> %s AS d FROM "{self.table}" '
            "WHERE NOT tombstoned ORDER BY d LIMIT %s",
            (np.asarray(vector, dtype=np.float32), k + 20),
        ).fetchall()
        return [h for h in (self._row_hit(r, float(r[4])) for r in rows) if not self.is_suppressed(h)][:k]

    def _filter_raw(self, key: str, value: Any, k: int) -> list[Hit]:
        rows = self._conn.execute(
            f'SELECT id, document, metadata, tombstoned FROM "{self.table}" '
            "WHERE metadata @> %s::jsonb LIMIT %s",
            (json.dumps({key: value}), k),
        ).fetchall()
        return [self._row_hit(r) for r in rows]

    def _mark_suppressed(self, keys: Sequence[str]) -> None:
        if keys:
            self._conn.execute(
                f'UPDATE "{self.table}" SET tombstoned = TRUE, '
                "metadata = metadata || %s::jsonb WHERE id = ANY(%s)",
                (json.dumps({K_SUPPRESSED: True}), list(keys)),
            )

    def _native_delete(self, keys: Sequence[str]) -> str:
        if keys:
            self._conn.execute(f'DELETE FROM "{self.table}" WHERE id = ANY(%s)', (list(keys),))
        return "DELETE (dead tuples remain until VACUUM; index entries until REINDEX)"

    def _vector_of(self, key: str) -> list[float] | None:
        row = self._conn.execute(
            f'SELECT embedding FROM "{self.table}" WHERE id = %s', (key,)
        ).fetchone()
        if row is None:
            return None
        return [float(x) for x in row[0]]

    def all_keys(self) -> list[str]:
        rows = self._conn.execute(f'SELECT id FROM "{self.table}"').fetchall()
        return [str(r[0]) for r in rows]

    def count(self) -> int:
        return int(self._conn.execute(f'SELECT COUNT(*) FROM "{self.table}"').fetchone()[0])

    def sample_keys(self, n: int) -> list[str]:
        rows = self._conn.execute(f'SELECT id FROM "{self.table}" ORDER BY id LIMIT %s', (n,)).fetchall()
        return [str(r[0]) for r in rows]

    def persisted_files(self) -> list[Path]:
        return []  # read through the server, see _relation_bytes

    # --- reclaim ---------------------------------------------------------------------------------

    def _reclaim(self, keys: Sequence[str]) -> ReclaimResult:
        present = self._get(keys)
        to_delete = [k for k in keys if k in present]
        if to_delete:
            self._native_delete(to_delete)
        if self.maintenance != "owner":
            return ReclaimResult(
                noop=not to_delete,
                method="DELETE only (REINDEX/VACUUM not permitted: not table owner)",
                measurement={"deleted": float(len(to_delete)), "vacuum_full": 0.0},
                detail="run REINDEX + VACUUM FULL from an owner role, then `tombstone verify`",
            )
        method = "DELETE + REINDEX INDEX + VACUUM FULL"
        full = 1.0
        self._conn.execute(f'REINDEX INDEX "{self.index_name}"')
        try:
            self._conn.execute(f'VACUUM FULL "{self.table}"')
        except Exception:  # noqa: BLE001
            self._conn.execute(f'VACUUM "{self.table}"')
            self._conn.execute(f'REINDEX INDEX "{self.index_name}"')
            method = "DELETE + VACUUM + REINDEX (VACUUM FULL denied)"
            full = 0.0
        if self.is_superuser:
            self._conn.execute("CHECKPOINT")
        return ReclaimResult(
            noop=not to_delete,
            method=method,
            measurement={"deleted": float(len(to_delete)), "vacuum_full": full},
        )

    # --- physical --------------------------------------------------------------------------------

    def _relation_files(self) -> list[tuple[str, str]]:
        """(label, server path) for the heap, its TOAST table, and the HNSW index (all forks/segments)."""
        rels = [self.table, self.index_name]
        row = self._conn.execute(
            "SELECT reltoastrelid::regclass::text FROM pg_class WHERE oid = %s::regclass AND reltoastrelid <> 0",
            (self.table,),
        ).fetchone()
        if row and row[0]:
            rels.append(str(row[0]))
        out: list[tuple[str, str]] = []
        for rel in rels:
            path = self._conn.execute("SELECT pg_relation_filepath(%s::regclass)", (rel,)).fetchone()[0]
            size = int(self._conn.execute("SELECT pg_relation_size(%s::regclass)", (rel,)).fetchone()[0])
            out.append((rel, str(path)))
            seg = 1
            while size > seg * (1 << 30):
                out.append((f"{rel}.{seg}", f"{path}.{seg}"))
                seg += 1
        return out

    def _relation_bytes(self, server_path: str) -> bytes:
        row = self._conn.execute("SELECT pg_read_binary_file(%s)", (server_path,)).fetchone()
        return bytes(row[0]) if row and row[0] is not None else b""

    def probe_physical(self, ref: ArtifactRef) -> PhysicalProbeResult:
        if VerifyLevel.PHYSICAL not in self.capabilities:
            raise NotSupported(
                f"store {self.name!r} cannot be physically verified: {self.physical_unsupported_reason()}"
            )
        patterns: dict[str, bytes] = {"artifact_id": ref.artifact_id.encode("utf-8")}
        if ref.embedding_fingerprint:
            patterns["f32le"] = fingerprint_bytes(ref.embedding_fingerprint)
        if self.is_superuser:
            self._conn.execute("CHECKPOINT")  # flush dirty pages so the file reflects the heap
        measurement: dict[str, float] = {}
        locations: list[str] = []
        total = 0
        for label, spath in self._relation_files():
            data = self._relation_bytes(spath)
            measurement[f"bytes_{label}"] = float(len(data))
            counts = {name: data.count(pat) for name, pat in patterns.items()}
            hit = sum(counts.values())
            if hit:
                locations.append(f"{label}:{'+'.join(k for k, v in counts.items() if v)}")
                total += hit
        method = "heap+index file scan via pg_read_binary_file"
        if self.pgstattuple:
            dead = self._conn.execute(f"SELECT dead_tuple_count FROM pgstattuple(%s)", (self.table,)).fetchone()[0]
            measurement["dead_tuples"] = float(dead)
            method = "pgstattuple + " + method
        measurement["matches"] = float(total)
        return PhysicalProbeResult(
            found=total > 0,
            method=method,
            locations=tuple(locations),
            measurement=measurement,
            detail=f"{total} match(es)" if total else "no match in heap, toast or index",
        )

    def persisted_scan_local(self, patterns: dict[str, bytes]) -> dict[str, int]:
        """For a co-located server: scan files directly (used by the backup experiment)."""
        out: dict[str, int] = {}
        for label, spath in self._relation_files():
            p = Path(spath)
            if p.is_file():
                out[label] = sum(scan_file(p, patterns).values())
        return out
