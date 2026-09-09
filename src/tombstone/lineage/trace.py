"""``trace()``: the pure reachability core (Hard Rules 3, 4, 8, 9).

Given a subject, a scope and an immutable ``LineageSnapshot``, return every artifact that
descends from the subject's SOURCE nodes. No I/O, no clock, no randomness, no network, no LLM.
Same (subject, scope, snapshot) → byte-identical ``Trace``, forever: the trace id is derived
from the inputs, artifacts are sorted, and every collection is ordered.
"""

from __future__ import annotations

from collections import deque

from tombstone.errors import LineageGapError, ScopeViolation
from tombstone.model.artifacts import ArtifactKind, ArtifactRef, Scope, SubjectRef
from tombstone.model.lineage import Edge, LineageSnapshot, Node, Trace
from tombstone.util import derived_ulid

_STRUCTURAL_VIAS = frozenset({"derived_from"})


def _check_scopes(snapshot: LineageSnapshot, nodes: dict[str, Node]) -> None:
    for e in snapshot.edges:
        p, c = nodes.get(e.parent), nodes.get(e.child)
        if p is None or c is None:
            continue
        if p.scope != c.scope:
            raise ScopeViolation(
                f"edge {e.parent} → {e.child} (via {e.via}) crosses scopes "
                f"{p.scope.tenant!r} → {c.scope.tenant!r}; refusing to trace (Hard Rule 8)"
            )


def _descendants(roots: set[str], children: dict[str, list[Edge]]) -> tuple[set[str], list[Edge]]:
    seen: set[str] = set(roots)
    used: list[Edge] = []
    queue: deque[str] = deque(sorted(roots))
    while queue:
        cur = queue.popleft()
        for e in children.get(cur, ()):
            used.append(e)
            if e.child not in seen:
                seen.add(e.child)
                queue.append(e.child)
    return seen, used


def _has_source_ancestor(
    node_id: str, parents: dict[str, list[Edge]], nodes: dict[str, Node]
) -> bool:
    seen: set[str] = set()
    stack = [node_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        n = nodes.get(cur)
        if n is not None and n.kind is ArtifactKind.SOURCE:
            return True
        stack.extend(e.parent for e in parents.get(cur, ()))
    return False


def trace(subject: SubjectRef, scope: Scope, graph: LineageSnapshot) -> Trace:
    """Pure function. No I/O, no clock, no randomness, no network, no LLM.

    Hard Rule 9: same (subject, scope, snapshot) MUST yield the same Trace, forever.
    Hard Rule 4: any reachable node without a path to a SOURCE, or any registered
    store with no capture hook, appears in ``gaps``. Non-empty gaps → the CLI refuses to erase.
    Hard Rule 8: an edge crossing scopes raises ScopeViolation.
    """
    if graph.scope != scope:
        raise ScopeViolation(
            f"snapshot is for scope {graph.scope.tenant!r}, trace requested for {scope.tenant!r}"
        )
    nodes = graph.node_map()
    _check_scopes(graph, nodes)

    children: dict[str, list[Edge]] = {}
    parents: dict[str, list[Edge]] = {}
    for e in graph.edges:
        children.setdefault(e.parent, []).append(e)
        parents.setdefault(e.child, []).append(e)

    owned = [n for n in graph.nodes if n.subject_hmac == subject.hmac and n.scope == scope]
    if not owned:
        raise LineageGapError(
            f"no lineage records for subject {subject.short} in scope {scope.tenant!r}: nothing "
            "was ever stamped for this subject, or it was ingested before capture was enabled. "
            "Refusing to report 'nothing to delete' (Hard Rule 4)."
        )
    roots = {n.artifact_id for n in owned if n.kind is ArtifactKind.SOURCE}
    reachable, used_edges = _descendants(roots, children)

    gaps: list[str] = []
    # (a) nodes stamped for this subject that no SOURCE reaches: a missing parent edge
    for n in owned:
        if n.artifact_id not in reachable and not _has_source_ancestor(
            n.artifact_id, parents, nodes
        ):
            gaps.append(
                f"orphan: {n.kind.value} {n.artifact_id} in {n.store} has no path to a SOURCE"
            )
    # (b) registered stores with no capture activity in this scope
    stores_seen = {n.store for n in graph.nodes}
    for s in graph.registered_stores:
        if s not in stores_seen:
            gaps.append(f"store {s}: registered but has no lineage nodes in scope {scope.tenant!r}")
    # (c) store contents that have no lineage node (measured by lineage.gaps before snapshot)
    for store, count in graph.store_gaps:
        if count < 0:
            gaps.append(f"store {store}: entries present with no lineage (count unknown)")
        elif count > 0:
            gaps.append(f"store {store}: ~{count} entries with no lineage node")

    artifacts: list[ArtifactRef] = []
    shared: list[str] = []
    for aid in sorted(reachable):
        reached = nodes.get(aid)
        if reached is None:
            gaps.append(f"dangling edge target {aid}: referenced by an edge but has no node")
            continue
        if reached.kind is ArtifactKind.SOURCE and reached.subject_hmac != subject.hmac:
            # reached another subject's SOURCE (e.g. via derived_from): that is theirs, not ours
            continue
        artifacts.append(reached.ref())
        if reached.subject_hmac != subject.hmac or _other_subject_parent(
            reached, parents, nodes, subject
        ):
            shared.append(aid)

    third_party = sorted(
        (
            nodes[m.source_artifact_id].ref()
            for m in graph.mentions
            if m.subject_hmac == subject.hmac and m.source_artifact_id in nodes
        ),
        key=lambda a: a.artifact_id,
    )
    already = sorted(a.artifact_id for a in artifacts if a.artifact_id in graph.tombstoned)
    snapshot_hash = graph.snapshot_hash()
    return Trace(
        trace_id=derived_ulid("trace", subject.hmac, scope.tenant, snapshot_hash),
        subject=subject,
        scope=scope,
        snapshot_hash=snapshot_hash,
        artifacts=tuple(artifacts),
        gaps=tuple(sorted(set(gaps))),
        third_party_hits=tuple(third_party),
        shared=tuple(sorted(set(shared))),
        already_tombstoned=tuple(already),
        edges=tuple(
            sorted(
                {e for e in used_edges if e.child in reachable},
                key=lambda e: (e.parent, e.child, e.via),
            )
        ),
        store_gaps=tuple(graph.store_gaps),
    )


def _other_subject_parent(
    node: Node, parents: dict[str, list[Edge]], nodes: dict[str, Node], subject: SubjectRef
) -> bool:
    """A chunk with a parent SOURCE owned by another subject is shared between subjects."""
    for e in parents.get(node.artifact_id, ()):
        p = nodes.get(e.parent)
        if p is not None and p.kind is ArtifactKind.SOURCE and p.subject_hmac != subject.hmac:
            return True
    return False
