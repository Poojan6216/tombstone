"""The lineage graph: nodes, edges, an immutable snapshot, and the result of a trace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tombstone.model.artifacts import ArtifactKind, ArtifactRef, Scope, SubjectRef
from tombstone.util import canonical_json, sha256_hex


@dataclass(frozen=True, slots=True)
class Node:
    artifact_id: str
    kind: ArtifactKind
    store: str
    store_key: str
    scope: Scope
    content_hash: str
    embedding_fingerprint: str | None
    subject_hmac: str  # owner subject. For SOURCE this is the data subject; derived nodes inherit.
    created_seq: int

    def ref(self) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=self.artifact_id,
            kind=self.kind,
            store=self.store,
            store_key=self.store_key,
            scope=self.scope,
            content_hash=self.content_hash,
            embedding_fingerprint=self.embedding_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "store": self.store,
            "store_key": self.store_key,
            "scope": self.scope.tenant,
            "content_hash": self.content_hash,
            "embedding_fingerprint": self.embedding_fingerprint,
            "subject_hmac": self.subject_hmac,
            "created_seq": self.created_seq,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Node:
        return Node(
            artifact_id=str(d["artifact_id"]),
            kind=ArtifactKind(d["kind"]),
            store=str(d["store"]),
            store_key=str(d["store_key"]),
            scope=Scope(str(d["scope"])),
            content_hash=str(d["content_hash"]),
            embedding_fingerprint=(
                str(d["embedding_fingerprint"]) if d.get("embedding_fingerprint") else None
            ),
            subject_hmac=str(d["subject_hmac"]),
            created_seq=int(d["created_seq"]),
        )


@dataclass(frozen=True, slots=True)
class Edge:
    parent: str  # artifact_id
    child: str  # artifact_id
    via: str  # "chunk", "embed:all-MiniLM-L6-v2", "cache:semantic", "train:shard-7", ...

    def to_dict(self) -> dict[str, str]:
        return {"parent": self.parent, "child": self.child, "via": self.via}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Edge:
        return Edge(parent=str(d["parent"]), child=str(d["child"]), via=str(d["via"]))


@dataclass(frozen=True, slots=True)
class Mention:
    """``SOURCE(other subject) → mentions → SUBJECT(this)``. Recorded, never searched for."""

    source_artifact_id: str
    subject_hmac: str  # the mentioned subject

    def to_dict(self) -> dict[str, str]:
        return {"source_artifact_id": self.source_artifact_id, "subject_hmac": self.subject_hmac}


@dataclass(frozen=True, slots=True)
class LineageSnapshot:
    """Everything ``trace()`` is allowed to look at. Sorted, hashable, scope-bounded.

    ``store_gaps`` is the count of store entries with no lineage node, per registered store, as
    measured by ``lineage.gaps`` (I/O) *before* the snapshot is handed to the pure trace.
    """

    scope: Scope
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    tombstoned: frozenset[str]  # artifact ids with a tombstone row
    mentions: tuple[Mention, ...]
    registered_stores: tuple[str, ...]
    store_gaps: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(sorted(self.nodes, key=lambda n: n.artifact_id)))
        object.__setattr__(
            self, "edges", tuple(sorted(self.edges, key=lambda e: (e.parent, e.child, e.via)))
        )
        object.__setattr__(
            self,
            "mentions",
            tuple(sorted(self.mentions, key=lambda m: (m.source_artifact_id, m.subject_hmac))),
        )
        object.__setattr__(self, "registered_stores", tuple(sorted(self.registered_stores)))
        object.__setattr__(self, "store_gaps", tuple(sorted(self.store_gaps)))

    def snapshot_hash(self) -> str:
        """Hash of the graph structure. Tombstones are deliberately excluded: suppression writes
        them mid-saga, and a resumed saga must still recognise its own trace as current."""
        payload = {
            "scope": self.scope.tenant,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "mentions": [m.to_dict() for m in self.mentions],
            "registered_stores": list(self.registered_stores),
            "store_gaps": [list(g) for g in self.store_gaps],
        }
        return sha256_hex(canonical_json(payload))

    def node_map(self) -> dict[str, Node]:
        return {n.artifact_id: n for n in self.nodes}


@dataclass(frozen=True, slots=True)
class Trace:
    trace_id: str  # ULID-shaped, derived from (subject, scope, snapshot_hash) — Hard Rule 9
    subject: SubjectRef
    scope: Scope
    snapshot_hash: str  # SHA-256 over the (sorted) node+edge set at trace time
    artifacts: tuple[ArtifactRef, ...]
    gaps: tuple[str, ...]  # Hard Rule 4
    third_party_hits: tuple[ArtifactRef, ...]  # SOURCE artifacts of OTHER subjects. NEEDS_HUMAN.
    shared: tuple[str, ...] = ()  # artifact ids also reachable from another subject's SOURCE
    already_tombstoned: tuple[str, ...] = ()  # artifact ids that carried a tombstone at trace time
    edges: tuple[Edge, ...] = field(default=())  # the edges within the traced set, for rendering
    store_gaps: tuple[tuple[str, int], ...] = ()  # (store, unlineaged count) measured at trace time

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "subject": self.subject.hmac,
            "scope": self.scope.tenant,
            "snapshot_hash": self.snapshot_hash,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "gaps": list(self.gaps),
            "third_party_hits": [a.to_dict() for a in self.third_party_hits],
            "shared": list(self.shared),
            "already_tombstoned": list(self.already_tombstoned),
            "edges": [e.to_dict() for e in self.edges],
            "store_gaps": [[str(k), int(v)] for k, v in self.store_gaps],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Trace:
        return Trace(
            trace_id=str(d["trace_id"]),
            subject=SubjectRef(str(d["subject"])),
            scope=Scope(str(d["scope"])),
            snapshot_hash=str(d["snapshot_hash"]),
            artifacts=tuple(ArtifactRef.from_dict(a) for a in d["artifacts"]),
            gaps=tuple(str(g) for g in d.get("gaps", [])),
            third_party_hits=tuple(ArtifactRef.from_dict(a) for a in d.get("third_party_hits", [])),
            shared=tuple(str(s) for s in d.get("shared", [])),
            already_tombstoned=tuple(str(s) for s in d.get("already_tombstoned", [])),
            edges=tuple(Edge.from_dict(e) for e in d.get("edges", [])),
            store_gaps=tuple((str(k), int(v)) for k, v in d.get("store_gaps", [])),
        )

    def by_store(self) -> dict[str, list[ArtifactRef]]:
        out: dict[str, list[ArtifactRef]] = {}
        for a in self.artifacts:
            out.setdefault(a.store, []).append(a)
        return out
