"""Hand-built lineage graphs for trace tests: a tiny DSL → LineageSnapshot."""

from __future__ import annotations

from tombstone.model.artifacts import ArtifactKind, Scope
from tombstone.model.lineage import Edge, LineageSnapshot, Mention, Node

KIND = {
    "S": ArtifactKind.SOURCE,
    "C": ArtifactKind.CHUNK,
    "E": ArtifactKind.EMBED,
    "K": ArtifactKind.CACHE,
    "T": ArtifactKind.TRAIN,
    "A": ArtifactKind.ADAPTER,
    "M": ArtifactKind.MEMORY,
}


def node(
    aid: str, subject: str = "s1", scope: str = "default", store: str | None = None, seq: int = 0
) -> Node:
    kind = KIND[aid[0]]
    return Node(
        artifact_id=aid,
        kind=kind,
        store=store or f"store-{kind.value}",
        store_key=f"key-{aid}",
        scope=Scope(scope),
        content_hash="h" * 64,
        embedding_fingerprint="00" * 128 if kind is ArtifactKind.EMBED else None,
        subject_hmac=subject,
        created_seq=seq,
    )


def graph(
    nodes: list[Node],
    edges: list[tuple[str, str, str]] = (),
    scope: str = "default",
    tombstoned: set[str] = frozenset(),
    mentions: list[tuple[str, str]] = (),
    registered: tuple[str, ...] = (),
    store_gaps: tuple[tuple[str, int], ...] = (),
) -> LineageSnapshot:
    return LineageSnapshot(
        scope=Scope(scope),
        nodes=tuple(nodes),
        edges=tuple(Edge(p, c, v) for p, c, v in edges),
        tombstoned=frozenset(tombstoned),
        mentions=tuple(Mention(src, subj) for src, subj in mentions),
        registered_stores=registered or tuple(sorted({n.store for n in nodes})),
        store_gaps=store_gaps,
    )
