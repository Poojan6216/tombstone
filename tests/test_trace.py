"""2.1–2.3: the pure trace — determinism (hypothesis), golden graphs, scope isolation, mentions."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests._graphs import graph, node
from tombstone.errors import LineageGapError, ScopeViolation
from tombstone.lineage.trace import trace
from tombstone.model.artifacts import ArtifactKind, Scope, SubjectRef
from tombstone.model.lineage import Edge, LineageSnapshot, Node

S1, S2 = SubjectRef("s1"), SubjectRef("s2")
D = Scope("default")


def ids(t) -> set[str]:
    return {a.artifact_id for a in t.artifacts}


# --- golden graphs -------------------------------------------------------------------------------

GOLDEN: list[tuple[str, LineageSnapshot, set[str], dict]] = []


def golden(name: str, snap: LineageSnapshot, expect: set[str], **extra) -> None:
    GOLDEN.append((name, snap, expect, extra))


golden("single source", graph([node("S1")]), {"S1"})
golden(
    "source→chunk→embed",
    graph([node("S1"), node("C1"), node("E1")], [("S1", "C1", "chunk"), ("C1", "E1", "embed:m")]),
    {"S1", "C1", "E1"},
)
golden(
    "two sources same subject",
    graph(
        [node("S1"), node("S2"), node("C1"), node("C2")],
        [("S1", "C1", "chunk"), ("S2", "C2", "chunk")],
    ),
    {"S1", "S2", "C1", "C2"},
)
golden(
    "diamond: one chunk in two indexes and one cache",
    graph(
        [node("S1"), node("C1"), node("E1", store="chroma"), node("E2", store="pg"), node("K1")],
        [
            ("S1", "C1", "chunk"),
            ("C1", "E1", "embed:a"),
            ("C1", "E2", "embed:b"),
            ("C1", "K1", "cache:semantic"),
        ],
    ),
    {"S1", "C1", "E1", "E2", "K1"},
)
golden(
    "other subject's data excluded",
    graph(
        [node("S1"), node("C1"), node("S2", subject="s2"), node("C2", subject="s2")],
        [("S1", "C1", "chunk"), ("S2", "C2", "chunk")],
    ),
    {"S1", "C1"},
)
golden(
    "shared chunk appears for both subjects and is flagged",
    graph(
        [node("S1"), node("S2", subject="s2"), node("C1")],
        [("S1", "C1", "chunk"), ("S2", "C1", "chunk")],
    ),
    {"S1", "C1"},
    shared={"C1"},
)
golden(
    "deep chain to train and adapter",
    graph(
        [node("S1"), node("C1"), node("T1"), node("A1")],
        [("S1", "C1", "chunk"), ("C1", "T1", "train:shard-3"), ("T1", "A1", "adapter")],
    ),
    {"S1", "C1", "T1", "A1"},
)
golden(
    "cycle-free repeated edges (idempotent)",
    graph([node("S1"), node("C1")], [("S1", "C1", "chunk"), ("S1", "C1", "chunk")]),
    {"S1", "C1"},
)
golden("orphan embed stamped for subject → gap", graph([node("S1"), node("E9")]), {"S1"}, gaps=1)
golden(
    "registered store with no nodes → gap",
    graph([node("S1")], registered=("store-source", "chroma:kb")),
    {"S1"},
    gaps=1,
)
golden(
    "store gap count propagates",
    graph([node("S1")], store_gaps=(("chroma:kb", 20),)),
    {"S1"},
    gaps=1,
)
golden(
    "store gap unknown count", graph([node("S1")], store_gaps=(("chroma:kb", -1),)), {"S1"}, gaps=1
)
golden(
    "mention recorded as third party, not target",
    graph([node("S1"), node("S2", subject="s2")], mentions=[("S2", "s1")]),
    {"S1"},
    third_party={"S2"},
)
golden(
    "mention of someone else ignored",
    graph([node("S1"), node("S2", subject="s2")], mentions=[("S1", "s2")]),
    {"S1"},
    third_party=set(),
)
golden(
    "derived_from reaches derived source of the same subject",
    graph([node("S1"), node("S2")], [("S1", "S2", "derived_from")]),
    {"S1", "S2"},
)
golden(
    "derived_from into another subject's source is not a target",
    graph(
        [node("S1"), node("S2", subject="s2"), node("C2", subject="s2")],
        [("S1", "S2", "derived_from"), ("S2", "C2", "chunk")],
    ),
    {"S1", "C2"},
    shared={"C2"},
)
golden(
    "already tombstoned artifacts listed",
    graph([node("S1"), node("C1")], [("S1", "C1", "chunk")], tombstoned={"C1"}),
    {"S1", "C1"},
    tombstoned={"C1"},
)
golden(
    "wide fan-out",
    graph(
        [node("S1")] + [node(f"C{i}") for i in range(10)] + [node(f"E{i}") for i in range(10)],
        [("S1", f"C{i}", "chunk") for i in range(10)]
        + [(f"C{i}", f"E{i}", "embed:m") for i in range(10)],
    ),
    {"S1"} | {f"C{i}" for i in range(10)} | {f"E{i}" for i in range(10)},
)
golden(
    "cache from two subjects' chunks (one node per subject)",
    graph(
        [
            node("S1"),
            node("C1"),
            node("K1"),
            node("S2", subject="s2"),
            node("C2", subject="s2"),
            node("K2", subject="s2"),
        ],
        [
            ("S1", "C1", "chunk"),
            ("C1", "K1", "cache:exact"),
            ("S2", "C2", "chunk"),
            ("C2", "K2", "cache:exact"),
        ],
    ),
    {"S1", "C1", "K1"},
)
golden("edge to missing node → gap", graph([node("S1")], [("S1", "C404", "chunk")]), {"S1"}, gaps=1)
golden("memory entries", graph([node("S1"), node("M1")], [("S1", "M1", "memory")]), {"S1", "M1"})
golden(
    "six deep",
    graph(
        [node("S1"), node("C1"), node("E1"), node("K1"), node("T1"), node("A1"), node("M1")],
        [
            ("S1", "C1", "chunk"),
            ("C1", "E1", "embed:m"),
            ("E1", "K1", "cache:semantic"),
            ("K1", "T1", "train:shard-0"),
            ("T1", "A1", "adapter"),
            ("A1", "M1", "memory"),
        ],
    ),
    {"S1", "C1", "E1", "K1", "T1", "A1", "M1"},
)
golden(
    "subject with only chunks (no source) → orphan gaps",
    graph([node("S9", subject="s2"), node("C1"), node("C2")], []),
    set(),
    gaps=2,
)
golden(
    "chunk of other subject reachable via shared source edge",
    graph([node("S1"), node("C7", subject="s2")], [("S1", "C7", "chunk")]),
    {"S1", "C7"},
    shared={"C7"},
)
golden(
    "two scopes present, only ours traced",
    graph(
        [
            node("S1"),
            node("C1"),
            node("S3", subject="s3"),
            node("C3", subject="s3"),
        ],
        [("S1", "C1", "chunk"), ("S3", "C3", "chunk")],
    ),
    {"S1", "C1"},
)

golden(
    "tombstoned parent still traversed",
    graph(
        [node("S1"), node("C1"), node("E1")],
        [("S1", "C1", "chunk"), ("C1", "E1", "embed:m")],
        tombstoned={"C1"},
    ),
    {"S1", "C1", "E1"},
    tombstoned={"C1"},
)
golden(
    "multiple mentions",
    graph(
        [node("S1"), node("S2", subject="s2"), node("S3", subject="s3")],
        mentions=[("S2", "s1"), ("S3", "s1")],
    ),
    {"S1"},
    third_party={"S2", "S3"},
)
golden(
    "mention of self ignored as target",
    graph([node("S1"), node("S2")], mentions=[("S2", "s1")]),
    {"S1", "S2"},
    third_party={"S2"},
)
golden(
    "embeds in three stores",
    graph(
        [
            node("S1"),
            node("C1"),
            node("E1", store="a"),
            node("E2", store="b"),
            node("E3", store="c"),
        ],
        [
            ("S1", "C1", "chunk"),
            ("C1", "E1", "embed:x"),
            ("C1", "E2", "embed:y"),
            ("C1", "E3", "embed:z"),
        ],
    ),
    {"S1", "C1", "E1", "E2", "E3"},
)
golden(
    "unrelated orphan of another subject is not our gap",
    graph([node("S1"), node("E5", subject="s2")]),
    {"S1"},
    gaps=0,
)

assert len(GOLDEN) >= 30


@pytest.mark.parametrize("name,snap,expect,extra", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_golden(name: str, snap: LineageSnapshot, expect: set[str], extra: dict) -> None:
    t = trace(S1, D, snap)
    assert ids(t) == expect, name
    if "shared" in extra:
        assert set(t.shared) == extra["shared"], name
    if "third_party" in extra:
        assert {a.artifact_id for a in t.third_party_hits} == extra["third_party"], name
    if "gaps" in extra:
        assert len(t.gaps) == extra["gaps"], (name, t.gaps)
    if "tombstoned" in extra:
        assert set(t.already_tombstoned) == extra["tombstoned"], name
    # every artifact is in scope; sorted; trace id derived
    assert all(a.scope == D for a in t.artifacts)
    assert [a.artifact_id for a in t.artifacts] == sorted(a.artifact_id for a in t.artifacts)
    assert t.snapshot_hash == snap.snapshot_hash()
    assert trace(S1, D, snap) == t  # deterministic


def test_shared_chunk_in_both_traces() -> None:
    snap = graph(
        [node("S1"), node("S2", subject="s2"), node("C1")],
        [("S1", "C1", "chunk"), ("S2", "C1", "chunk")],
    )
    assert "C1" in ids(trace(S1, D, snap)) and "C1" in ids(trace(S2, D, snap))
    assert trace(S1, D, snap).shared == ("C1",) and trace(S2, D, snap).shared == ("C1",)


def test_no_lineage_raises_gap_error() -> None:
    snap = graph([node("S2", subject="s2")])
    with pytest.raises(LineageGapError, match="no lineage records"):
        trace(S1, D, snap)


# --- 2.2 scope isolation ------------------------------------------------------------------------


def test_cross_scope_edge_raises_naming_both_ids() -> None:
    snap = graph([node("S1"), node("C1", scope="tenant-b")], [("S1", "C1", "chunk")])
    with pytest.raises(ScopeViolation) as ei:
        trace(S1, D, snap)
    msg = str(ei.value)
    assert "S1" in msg and "C1" in msg and "tenant-b" in msg


def test_snapshot_scope_mismatch_raises() -> None:
    snap = graph([node("S1")], scope="other")
    with pytest.raises(ScopeViolation):
        trace(S1, D, snap)


@st.composite
def multi_scope_graphs(draw: st.DrawFn) -> tuple[LineageSnapshot, str]:
    scopes = ["a", "b", "c"]
    n = draw(st.integers(min_value=1, max_value=40))
    nodes: list[Node] = []
    for i in range(n):
        sc = draw(st.sampled_from(scopes))
        kind = draw(st.sampled_from("SCEKT"))
        subj = draw(st.sampled_from(["s1", "s2"]))
        nodes.append(node(f"{kind}{i}", subject=subj, scope=sc))
    edges: list[tuple[str, str, str]] = []
    m = draw(st.integers(min_value=0, max_value=n * 2))
    for _ in range(m):
        i = draw(st.integers(min_value=0, max_value=n - 1))
        j = draw(st.integers(min_value=0, max_value=n - 1))
        if i < j and nodes[i].scope == nodes[j].scope:  # no cross edges, acyclic
            edges.append((nodes[i].artifact_id, nodes[j].artifact_id, "e"))
    target = draw(st.sampled_from(scopes))
    in_scope = [x for x in nodes if x.scope.tenant == target]
    snap = LineageSnapshot(
        scope=Scope(target),
        nodes=tuple(in_scope),
        edges=tuple(
            Edge(p, c, v) for p, c, v in edges if any(x.artifact_id == p for x in in_scope)
        ),
        tombstoned=frozenset(),
        mentions=(),
        registered_stores=(),
    )
    return snap, target


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=multi_scope_graphs())
def test_multi_scope_without_cross_edges_stays_in_one_scope(
    data: tuple[LineageSnapshot, str],
) -> None:
    snap, target = data
    for subj in (S1, S2):
        try:
            t = trace(subj, Scope(target), snap)
        except LineageGapError:
            continue
        assert {a.scope.tenant for a in t.artifacts} <= {target}


# --- 2.1 determinism: 3000 random triples traced twice, byte-identical --------------------------


@st.composite
def random_snapshots(draw: st.DrawFn) -> tuple[SubjectRef, Scope, LineageSnapshot]:
    n = draw(st.integers(min_value=1, max_value=60))
    subjects = ["s1", "s2", "s3"]
    nodes = [
        node(
            f"{draw(st.sampled_from('SSCEKTA'))}{i}", subject=draw(st.sampled_from(subjects)), seq=i
        )
        for i in range(n)
    ]
    edges: set[tuple[str, str, str]] = set()
    for _ in range(draw(st.integers(min_value=0, max_value=n * 3))):
        i = draw(st.integers(min_value=0, max_value=n - 1))
        j = draw(st.integers(min_value=0, max_value=n - 1))
        if i < j:
            edges.add(
                (
                    nodes[i].artifact_id,
                    nodes[j].artifact_id,
                    draw(st.sampled_from(["chunk", "embed:m", "cache:x", "train:shard-1"])),
                )
            )
    tomb = {
        nodes[i].artifact_id
        for i in draw(st.lists(st.integers(min_value=0, max_value=n - 1), max_size=5))
    }
    mentions = [
        (nodes[i].artifact_id, draw(st.sampled_from(subjects)))
        for i in draw(st.lists(st.integers(min_value=0, max_value=n - 1), max_size=3))
        if nodes[i].kind is ArtifactKind.SOURCE
    ]
    snap = graph(
        nodes,
        sorted(edges),
        tombstoned=tomb,
        mentions=mentions,
        store_gaps=tuple(
            (f"st{k}", draw(st.integers(min_value=-1, max_value=50)))
            for k in range(draw(st.integers(min_value=0, max_value=2)))
        ),
    )
    return SubjectRef(draw(st.sampled_from(subjects))), D, snap


@settings(
    max_examples=3000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(triple=random_snapshots())
def test_trace_is_deterministic(triple: tuple[SubjectRef, Scope, LineageSnapshot]) -> None:
    subject, scope, snap = triple
    try:
        a = trace(subject, scope, snap)
    except LineageGapError:
        with pytest.raises(LineageGapError):
            trace(subject, scope, snap)
        return
    b = trace(subject, scope, snap)
    assert a == b
    assert a.to_dict() == b.to_dict()
    import json

    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)
    # only the subject's own or shared artifacts; targets never chosen by similarity — the
    # graph is the only input (no content anywhere in the snapshot)
    for art in a.artifacts:
        n = snap.node_map()[art.artifact_id]
        assert (
            n.subject_hmac == subject.hmac
            or art.artifact_id in a.shared
            or any(e.child == art.artifact_id for e in a.edges)
        )
