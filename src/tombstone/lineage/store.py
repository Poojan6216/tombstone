"""The lineage store: SQLite by default, PostgreSQL optionally, one DDL for both.

Append-only on ``nodes`` and ``edges``. Every write that needs an ordering takes a sequence number
from ``counters`` inside the same transaction, so ``created_seq`` is a total order per database.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

from tombstone.config import LineageConfig, resolve_env
from tombstone.errors import ConfigError
from tombstone.model.artifacts import ArtifactKind, Scope, SubjectRef
from tombstone.model.lineage import Edge, LineageSnapshot, Mention, Node, Trace
from tombstone.model.pins import Pin, pin_from_dict
from tombstone.util import canonical_json

SCHEMA_VERSION = 1


def schema_sql() -> str:
    return resources.files("tombstone.lineage").joinpath("schema.sql").read_text(encoding="utf-8")


class _Cursor(Protocol):
    def fetchone(self) -> Any: ...
    def fetchall(self) -> list[Any]: ...


class _Dialect:
    """Translates the tiny SQL surface we use between sqlite3 and psycopg."""

    def __init__(self, backend: str) -> None:
        self.backend = backend

    def q(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.backend == "postgres" else sql

    @property
    def insert_ignore(self) -> str:
        return "INSERT INTO" if self.backend == "postgres" else "INSERT OR IGNORE INTO"

    @property
    def on_conflict_nothing(self) -> str:
        return " ON CONFLICT DO NOTHING" if self.backend == "postgres" else ""


class LineageStore:
    """One connection, explicit transactions, dialect-translated SQL."""

    def __init__(self, conn: Any, backend: str, location: str) -> None:
        self._conn = conn
        self._d = _Dialect(backend)
        self.backend = backend
        self.location = location
        self._depth = 0
        # One connection per store, shared across threads (sqlite3 check_same_thread=False):
        # every statement and every transaction is serialised by this re-entrant lock.
        self._lock = threading.RLock()
        self._init_schema()

    # --- construction ------------------------------------------------------------------------

    @classmethod
    def open(cls, cfg: LineageConfig) -> LineageStore:
        if cfg.backend == "sqlite":
            return cls.open_sqlite(cfg.path)
        dsn = resolve_env(cfg.dsn)
        if not dsn:
            raise ConfigError("lineage.dsn is required for the postgres backend")
        return cls.open_postgres(dsn)

    @classmethod
    def open_sqlite(cls, path: str | Path) -> LineageStore:
        p = Path(path)
        if str(p) != ":memory:":
            p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), isolation_level=None, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return cls(conn, "sqlite", str(p))

    @classmethod
    def open_postgres(cls, dsn: str) -> LineageStore:
        try:
            import psycopg
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ConfigError(
                "lineage.backend=postgres requires the [pgvector] extra: "
                "uv pip install 'tombstone-erase[pgvector]'"
            ) from e
        conn = psycopg.connect(dsn, autocommit=True)
        return cls(conn, "postgres", "postgres")

    def close(self) -> None:
        if self.backend == "sqlite":
            # fold the WAL into the main file so a copy of lineage.db is complete on its own
            with contextlib.suppress(Exception):
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._conn.close()

    # --- low level ---------------------------------------------------------------------------

    def _exec(self, sql: str, params: Sequence[Any] = ()) -> Any:
        with self._lock:
            return self._conn.execute(self._d.q(sql), tuple(params))

    def _executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        if self.backend == "sqlite":
            self._conn.executemany(self._d.q(sql), [tuple(r) for r in rows])
        else:
            with self._conn.cursor() as cur:
                cur.executemany(self._d.q(sql), [tuple(r) for r in rows])

    @contextmanager
    def tx(self) -> Iterator[None]:
        """Nested-safe transaction. Outermost BEGIN/COMMIT, inner calls are no-ops. The lock is
        held for the whole outermost transaction so two threads never interleave statements."""
        with self._lock:
            if self._depth == 0:
                self._exec("BEGIN")
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if self._depth == 0:
                    self._exec("ROLLBACK")
                raise
            else:
                self._depth -= 1
                if self._depth == 0:
                    self._exec("COMMIT")

    def _init_schema(self) -> None:
        ddl = "\n".join(
            line for line in schema_sql().splitlines() if not line.strip().startswith("--")
        )
        with self.tx():
            for stmt in ddl.split(";"):
                s = stmt.strip()
                if s:
                    self._exec(s)
            self._exec(
                f"{self._d.insert_ignore} meta (key, value) VALUES (?, ?)"
                f"{self._d.on_conflict_nothing}",
                ("schema_version", str(SCHEMA_VERSION)),
            )
            self._exec(
                f"{self._d.insert_ignore} counters (name, value) VALUES (?, ?)"
                f"{self._d.on_conflict_nothing}",
                ("seq", 0),
            )

    def next_seq(self) -> int:
        with self.tx():
            row = self._exec(
                "UPDATE counters SET value = value + 1 WHERE name = 'seq' RETURNING value"
            ).fetchone()
            return int(row[0])

    def current_seq(self) -> int:
        row = self._exec("SELECT value FROM counters WHERE name = 'seq'").fetchone()
        return int(row[0])

    # --- nodes & edges -------------------------------------------------------------------------

    def add_node(self, node: Node) -> bool:
        """Insert; returns False if the artifact id already existed (idempotent capture)."""
        with self.tx():
            cur = self._exec(
                f"{self._d.insert_ignore} nodes (artifact_id, kind, store, store_key, scope, "
                "content_hash, embedding_fingerprint, subject_hmac, created_seq) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?){self._d.on_conflict_nothing}",
                (
                    node.artifact_id,
                    node.kind.value,
                    node.store,
                    node.store_key,
                    node.scope.tenant,
                    node.content_hash,
                    node.embedding_fingerprint,
                    node.subject_hmac,
                    node.created_seq,
                ),
            )
            return bool(cur.rowcount)

    def add_nodes(self, nodes: Iterable[Node]) -> None:
        with self.tx():
            self._executemany(
                f"{self._d.insert_ignore} nodes (artifact_id, kind, store, store_key, scope, "
                "content_hash, embedding_fingerprint, subject_hmac, created_seq) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?){self._d.on_conflict_nothing}",
                (
                    (
                        n.artifact_id,
                        n.kind.value,
                        n.store,
                        n.store_key,
                        n.scope.tenant,
                        n.content_hash,
                        n.embedding_fingerprint,
                        n.subject_hmac,
                        n.created_seq,
                    )
                    for n in nodes
                ),
            )

    def add_edge(self, edge: Edge) -> None:
        with self.tx():
            self._exec(
                f"{self._d.insert_ignore} edges (parent, child, via) VALUES (?, ?, ?)"
                f"{self._d.on_conflict_nothing}",
                (edge.parent, edge.child, edge.via),
            )

    def add_edges(self, edges: Iterable[Edge]) -> None:
        with self.tx():
            self._executemany(
                f"{self._d.insert_ignore} edges (parent, child, via) VALUES (?, ?, ?)"
                f"{self._d.on_conflict_nothing}",
                ((e.parent, e.child, e.via) for e in edges),
            )

    def add_mention(self, source_artifact_id: str, mentioned: SubjectRef, scope: Scope) -> None:
        with self.tx():
            self._exec(
                f"{self._d.insert_ignore} mentions (source_artifact_id, subject_hmac, scope) "
                f"VALUES (?, ?, ?){self._d.on_conflict_nothing}",
                (source_artifact_id, mentioned.hmac, scope.tenant),
            )

    def node(self, artifact_id: str) -> Node | None:
        row = self._exec(
            "SELECT artifact_id, kind, store, store_key, scope, content_hash, "
            "embedding_fingerprint, subject_hmac, created_seq FROM nodes WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        return _row_to_node(row) if row else None

    def nodes(self, artifact_ids: Sequence[str]) -> list[Node]:
        out: list[Node] = []
        for i in range(0, len(artifact_ids), 500):
            batch = list(artifact_ids[i : i + 500])
            marks = ",".join("?" for _ in batch)
            rows = self._exec(
                "SELECT artifact_id, kind, store, store_key, scope, content_hash, "
                f"embedding_fingerprint, subject_hmac, created_seq FROM nodes WHERE artifact_id IN ({marks})",
                batch,
            ).fetchall()
            out.extend(_row_to_node(r) for r in rows)
        return out

    def nodes_for_subject(self, subject: SubjectRef, scope: Scope) -> list[Node]:
        rows = self._exec(
            "SELECT artifact_id, kind, store, store_key, scope, content_hash, "
            "embedding_fingerprint, subject_hmac, created_seq FROM nodes "
            "WHERE subject_hmac = ? AND scope = ? ORDER BY artifact_id",
            (subject.hmac, scope.tenant),
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def node_by_store_key(self, store: str, store_key: str) -> Node | None:
        row = self._exec(
            "SELECT artifact_id, kind, store, store_key, scope, content_hash, "
            "embedding_fingerprint, subject_hmac, created_seq FROM nodes "
            "WHERE store = ? AND store_key = ? ORDER BY created_seq DESC",
            (store, store_key),
        ).fetchone()
        return _row_to_node(row) if row else None

    def store_keys_present(self, store: str, keys: Sequence[str]) -> set[str]:
        present: set[str] = set()
        for i in range(0, len(keys), 500):
            batch = list(keys[i : i + 500])
            marks = ",".join("?" for _ in batch)
            rows = self._exec(
                f"SELECT store_key FROM nodes WHERE store = ? AND store_key IN ({marks})",
                [store, *batch],
            ).fetchall()
            present.update(str(r[0]) for r in rows)
        return present

    def edges_from(self, parents: Sequence[str]) -> list[Edge]:
        out: list[Edge] = []
        for i in range(0, len(parents), 500):
            batch = list(parents[i : i + 500])
            marks = ",".join("?" for _ in batch)
            rows = self._exec(
                f"SELECT parent, child, via FROM edges WHERE parent IN ({marks})", batch
            ).fetchall()
            out.extend(Edge(str(r[0]), str(r[1]), str(r[2])) for r in rows)
        return out

    def edges_to(self, children: Sequence[str]) -> list[Edge]:
        out: list[Edge] = []
        for i in range(0, len(children), 500):
            batch = list(children[i : i + 500])
            marks = ",".join("?" for _ in batch)
            rows = self._exec(
                f"SELECT parent, child, via FROM edges WHERE child IN ({marks})", batch
            ).fetchall()
            out.extend(Edge(str(r[0]), str(r[1]), str(r[2])) for r in rows)
        return out

    def live_duplicates(
        self, store: str, fingerprint: str | None, content_hash: str, exclude: str
    ) -> int:
        """Other non-tombstoned nodes in ``store`` holding the same vector bytes or content.
        Their bytes are indistinguishable from the artifact's own in a byte scan."""
        if fingerprint:
            row = self._exec(
                "SELECT COUNT(*) FROM nodes n WHERE n.store = ? AND n.artifact_id <> ? "
                "AND (n.embedding_fingerprint = ? OR n.content_hash = ?) "
                "AND n.artifact_id NOT IN (SELECT artifact_id FROM tombstones)",
                (store, exclude, fingerprint, content_hash),
            ).fetchone()
        else:
            row = self._exec(
                "SELECT COUNT(*) FROM nodes n WHERE n.store = ? AND n.artifact_id <> ? "
                "AND n.content_hash = ? "
                "AND n.artifact_id NOT IN (SELECT artifact_id FROM tombstones)",
                (store, exclude, content_hash),
            ).fetchone()
        return int(row[0]) if row else 0

    def reachable(self, roots: Sequence[str]) -> set[str]:
        """Descendants of ``roots`` (inclusive) via a recursive CTE. Used by tests and status."""
        if not roots:
            return set()
        marks = ",".join("?" for _ in roots)
        rows = self._exec(
            "WITH RECURSIVE reach(id) AS ("
            f"  SELECT artifact_id FROM nodes WHERE artifact_id IN ({marks})"
            "  UNION"
            "  SELECT e.child FROM edges e JOIN reach r ON e.parent = r.id"
            ") SELECT id FROM reach",
            list(roots),
        ).fetchall()
        return {str(r[0]) for r in rows}

    # --- tombstones --------------------------------------------------------------------------

    def tombstone(self, artifact_ids: Iterable[str], reason: str, trace_id: str | None) -> int:
        """Mark artifacts deleted. Returns the seq used. Idempotent (existing rows untouched)."""
        ids = list(artifact_ids)
        with self.tx():
            seq = self.next_seq()
            self._executemany(
                f"{self._d.insert_ignore} tombstones (artifact_id, tombstoned_seq, trace_id, reason) "
                f"VALUES (?, ?, ?, ?){self._d.on_conflict_nothing}",
                ((a, seq, trace_id, reason) for a in ids),
            )
            return seq

    def tombstoned_ids(self, scope: Scope) -> frozenset[str]:
        rows = self._exec(
            "SELECT t.artifact_id FROM tombstones t JOIN nodes n ON n.artifact_id = t.artifact_id "
            "WHERE n.scope = ?",
            (scope.tenant,),
        ).fetchall()
        return frozenset(str(r[0]) for r in rows)

    def tombstone_info(self, artifact_id: str) -> tuple[int, str | None, str] | None:
        row = self._exec(
            "SELECT tombstoned_seq, trace_id, reason FROM tombstones WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        return (int(row[0]), row[1], str(row[2])) if row else None

    # --- stores registry ---------------------------------------------------------------------

    def register_store(self, name: str, kind: str, scope: Scope) -> None:
        with self.tx():
            seq = self.next_seq()
            self._exec(
                f"{self._d.insert_ignore} stores (name, scope, kind, registered_seq) "
                f"VALUES (?, ?, ?, ?){self._d.on_conflict_nothing}",
                (name, scope.tenant, kind, seq),
            )

    def registered_stores(self, scope: Scope) -> list[tuple[str, str]]:
        rows = self._exec(
            "SELECT name, kind FROM stores WHERE scope = ? ORDER BY name", (scope.tenant,)
        ).fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    def counts_by_store(self, scope: Scope) -> dict[str, int]:
        rows = self._exec(
            "SELECT store, COUNT(*) FROM nodes WHERE scope = ? GROUP BY store", (scope.tenant,)
        ).fetchall()
        return {str(r[0]): int(r[1]) for r in rows}

    def counts_by_kind(self, scope: Scope) -> dict[str, int]:
        rows = self._exec(
            "SELECT kind, COUNT(*) FROM nodes WHERE scope = ? GROUP BY kind", (scope.tenant,)
        ).fetchall()
        return {str(r[0]): int(r[1]) for r in rows}

    def subject_count(self, scope: Scope) -> int:
        row = self._exec(
            "SELECT COUNT(DISTINCT subject_hmac) FROM nodes WHERE scope = ? AND kind = ?",
            (scope.tenant, ArtifactKind.SOURCE.value),
        ).fetchone()
        return int(row[0])

    # --- pins ------------------------------------------------------------------------------

    def put_pin(self, pin: Pin, reason: str) -> None:
        with self.tx():
            seq = self.next_seq()
            self._exec(
                "INSERT INTO pins (name, pinned_seq, payload, reason) VALUES (?, ?, ?, ?)",
                (pin.name, seq, canonical_json(pin.to_dict()), reason),
            )

    def current_pin(self, name: str) -> Pin | None:
        row = self._exec(
            "SELECT payload FROM pins WHERE name = ? ORDER BY pinned_seq DESC", (name,)
        ).fetchone()
        return pin_from_dict(json.loads(row[0])) if row else None

    def all_pins(self) -> dict[str, Pin]:
        rows = self._exec("SELECT name, payload FROM pins ORDER BY pinned_seq ASC").fetchall()
        out: dict[str, Pin] = {}
        for r in rows:
            out[str(r[0])] = pin_from_dict(json.loads(r[1]))
        return out

    # --- probes ------------------------------------------------------------------------------

    def put_probes(self, artifact_id: str, model: str, probes: Sequence[tuple[str, str]]) -> None:
        """``probes`` = (query_hash, embedding_hex) pairs. Query text is never stored."""
        with self.tx():
            self._executemany(
                f"{self._d.insert_ignore} probes (artifact_id, idx, query_hash, model, embedding_hex) "
                f"VALUES (?, ?, ?, ?, ?){self._d.on_conflict_nothing}",
                ((artifact_id, i, qh, model, eh) for i, (qh, eh) in enumerate(probes)),
            )

    def probes(self, artifact_id: str) -> list[tuple[str, str, str]]:
        rows = self._exec(
            "SELECT query_hash, model, embedding_hex FROM probes WHERE artifact_id = ? ORDER BY idx",
            (artifact_id,),
        ).fetchall()
        return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]

    # --- traces ------------------------------------------------------------------------------

    def save_trace(self, trace: Trace) -> None:
        with self.tx():
            seq = self.next_seq()
            self._exec(
                f"{self._d.insert_ignore} traces (trace_id, subject_hmac, scope, snapshot_hash, "
                f"payload, created_seq) VALUES (?, ?, ?, ?, ?, ?){self._d.on_conflict_nothing}",
                (
                    trace.trace_id,
                    trace.subject.hmac,
                    trace.scope.tenant,
                    trace.snapshot_hash,
                    canonical_json(trace.to_dict()),
                    seq,
                ),
            )

    def load_trace(self, trace_id: str) -> Trace | None:
        row = self._exec("SELECT payload FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
        return Trace.from_dict(json.loads(row[0])) if row else None

    # --- snapshot --------------------------------------------------------------------------------

    def snapshot(self, scope: Scope, store_gaps: Sequence[tuple[str, int]] = ()) -> LineageSnapshot:
        """Everything in ``scope``, sorted. Edges are included if either endpoint is in scope so
        that cross-scope edges are visible to the trace (which must raise on them)."""
        node_rows = self._exec(
            "SELECT artifact_id, kind, store, store_key, scope, content_hash, "
            "embedding_fingerprint, subject_hmac, created_seq FROM nodes WHERE scope = ?",
            (scope.tenant,),
        ).fetchall()
        nodes = [_row_to_node(r) for r in node_rows]
        edge_rows = self._exec(
            "SELECT e.parent, e.child, e.via FROM edges e "
            "WHERE e.parent IN (SELECT artifact_id FROM nodes WHERE scope = ?) "
            "   OR e.child IN (SELECT artifact_id FROM nodes WHERE scope = ?)",
            (scope.tenant, scope.tenant),
        ).fetchall()
        edges = [Edge(str(r[0]), str(r[1]), str(r[2])) for r in edge_rows]
        # Nodes on the far side of a cross-scope edge are loaded too, so trace can name them.
        in_scope = {n.artifact_id for n in nodes}
        foreign = sorted({e.parent for e in edges} | {e.child for e in edges})
        foreign = [f for f in foreign if f not in in_scope]
        if foreign:
            nodes.extend(self.nodes(foreign))
        mention_rows = self._exec(
            "SELECT source_artifact_id, subject_hmac FROM mentions WHERE scope = ?",
            (scope.tenant,),
        ).fetchall()
        mentions = [Mention(str(r[0]), str(r[1])) for r in mention_rows]
        return LineageSnapshot(
            scope=scope,
            nodes=tuple(nodes),
            edges=tuple(edges),
            tombstoned=self.tombstoned_ids(scope),
            mentions=tuple(mentions),
            registered_stores=tuple(n for n, _ in self.registered_stores(scope)),
            store_gaps=tuple(store_gaps),
        )


def _row_to_node(row: Sequence[Any]) -> Node:
    return Node(
        artifact_id=str(row[0]),
        kind=ArtifactKind(str(row[1])),
        store=str(row[2]),
        store_key=str(row[3]),
        scope=Scope(str(row[4])),
        content_hash=str(row[5]),
        embedding_fingerprint=str(row[6]) if row[6] else None,
        subject_hmac=str(row[7]),
        created_seq=int(row[8]),
    )
