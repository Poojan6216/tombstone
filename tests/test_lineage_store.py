from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import networkx as nx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tombstone.lineage.store import LineageStore
from tombstone.model.artifacts import ArtifactKind, Scope, SubjectRef
from tombstone.model.lineage import Edge, Node
from tombstone.model.pins import ManifestPin, StorePin
from tombstone.model.status import VerifyLevel


def _node(
    aid: str,
    kind: ArtifactKind = ArtifactKind.CHUNK,
    subject: str = "s1",
    scope: str = "default",
    seq: int = 1,
) -> Node:
    return Node(
        artifact_id=aid,
        kind=kind,
        store="test",
        store_key=f"k-{aid}",
        scope=Scope(scope),
        content_hash="h" * 64,
        embedding_fingerprint=None,
        subject_hmac=subject,
        created_seq=seq,
    )


@pytest.fixture(params=["sqlite", "postgres"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[LineageStore]:
    if request.param == "sqlite":
        s = LineageStore.open_sqlite(tmp_path / "lineage.db")
    else:
        dsn = request.getfixturevalue("pg_database")
        s = LineageStore.open_postgres(dsn)
    yield s
    s.close()


def test_schema_applies_and_seq_increments(store: LineageStore) -> None:
    assert store.current_seq() == 0
    assert store.next_seq() == 1
    assert store.next_seq() == 2
    assert store.current_seq() == 2


def test_nodes_and_edges_round_trip(store: LineageStore) -> None:
    a = _node("A", ArtifactKind.SOURCE)
    b = _node("B")
    assert store.add_node(a) is True
    assert store.add_node(a) is False  # idempotent
    store.add_nodes([b])
    store.add_edge(Edge("A", "B", "chunk"))
    store.add_edge(Edge("A", "B", "chunk"))  # idempotent
    assert store.node("A") == a
    assert store.node("Z") is None
    assert store.edges_from(["A"]) == [Edge("A", "B", "chunk")]
    assert store.edges_to(["B"]) == [Edge("A", "B", "chunk")]
    assert store.reachable(["A"]) == {"A", "B"}
    assert store.store_keys_present("test", ["k-A", "k-B", "k-Q"]) == {"k-A", "k-B"}
    assert store.node_by_store_key("test", "k-B") == b


def test_tombstones_are_separate_and_idempotent(store: LineageStore) -> None:
    store.add_nodes([_node("A", ArtifactKind.SOURCE), _node("B")])
    seq = store.tombstone(["A", "B"], reason="erase", trace_id="T1")
    seq2 = store.tombstone(["A"], reason="erase-again", trace_id="T2")
    assert seq2 > seq
    assert store.tombstoned_ids(Scope("default")) == {"A", "B"}
    info = store.tombstone_info("A")
    assert info is not None and info[0] == seq and info[1] == "T1" and info[2] == "erase"
    # nodes table untouched
    assert store.node("A") is not None


def test_mentions_and_stores_and_pins(store: LineageStore) -> None:
    store.add_node(_node("SRC", ArtifactKind.SOURCE, subject="other"))
    store.add_mention("SRC", SubjectRef("me"), Scope("default"))
    store.register_store("chroma:kb", "chroma", Scope("default"))
    store.register_store("chroma:kb", "chroma", Scope("default"))
    assert store.registered_stores(Scope("default")) == [("chroma:kb", "chroma")]
    snap = store.snapshot(Scope("default"))
    assert snap.mentions[0].source_artifact_id == "SRC" and snap.mentions[0].subject_hmac == "me"
    assert snap.registered_stores == ("chroma:kb",)
    pin = StorePin("chroma:kb", "chroma", "1.0", frozenset({VerifyLevel.LOGICAL}), "m", 384)
    store.put_pin(pin, "init")
    assert store.current_pin("chroma:kb") == pin
    pin2 = StorePin("chroma:kb", "chroma", "1.1", frozenset({VerifyLevel.LOGICAL}), "m", 384)
    store.put_pin(pin2, "repin: upgraded")
    assert store.current_pin("chroma:kb") == pin2
    store.put_pin(ManifestPin("ft", "abc", 10), "init")
    assert set(store.all_pins()) == {"chroma:kb", "ft"}


def test_probes_and_traces(store: LineageStore) -> None:
    store.put_probes("A", "minilm", [("qh1", "00ff"), ("qh2", "ff00")])
    assert store.probes("A") == [("qh1", "minilm", "00ff"), ("qh2", "minilm", "ff00")]
    assert store.load_trace("nope") is None


def test_snapshot_is_scope_bounded_but_sees_cross_scope_edges(store: LineageStore) -> None:
    store.add_nodes([_node("A", ArtifactKind.SOURCE, scope="a"), _node("B", scope="b")])
    store.add_edge(Edge("A", "B", "chunk"))
    snap = store.snapshot(Scope("a"))
    ids = {n.artifact_id for n in snap.nodes}
    assert ids == {"A", "B"}  # B is loaded so trace can name it in the ScopeViolation
    assert snap.edges == (Edge("A", "B", "chunk"),)
    assert snap.snapshot_hash() == store.snapshot(Scope("a")).snapshot_hash()


# --- hypothesis: recursive CTE reachability == NetworkX ---------------------------------------


@st.composite
def dags(draw: st.DrawFn) -> tuple[list[str], list[tuple[str, str]]]:
    n = draw(st.integers(min_value=1, max_value=120))
    ids = [f"N{i:04d}" for i in range(n)]
    edges: set[tuple[str, str]] = set()
    m = draw(st.integers(min_value=0, max_value=min(400, n * 3)))
    for _ in range(m):
        i = draw(st.integers(min_value=0, max_value=n - 1))
        j = draw(st.integers(min_value=0, max_value=n - 1))
        if i < j:  # parent index < child index ⇒ acyclic
            edges.add((ids[i], ids[j]))
    return ids, sorted(edges)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(dag=dags(), root_count=st.integers(min_value=1, max_value=5))
def test_reachability_matches_networkx_sqlite(
    tmp_path_factory: pytest.TempPathFactory,
    dag: tuple[list[str], list[tuple[str, str]]],
    root_count: int,
) -> None:
    ids, edges = dag
    path = tmp_path_factory.mktemp("hyp") / "lineage.db"
    s = LineageStore.open_sqlite(path)
    try:
        _check_reachability(s, ids, edges, root_count)
    finally:
        s.close()


@pytest.mark.pg
def test_reachability_matches_networkx_postgres(pg_database: str) -> None:
    # A smaller deterministic sample of DAGs on Postgres (hypothesis + session fixtures don't mix).
    import random

    rng = random.Random(1234)
    s = LineageStore.open_postgres(pg_database)
    try:
        for trial in range(6):
            n = rng.randint(1, 80)
            ids = [f"T{trial}N{i:04d}" for i in range(n)]
            edges = set()
            for _ in range(rng.randint(0, n * 2)):
                i, j = rng.randint(0, n - 1), rng.randint(0, n - 1)
                if i < j:
                    edges.add((ids[i], ids[j]))
            _check_reachability(s, ids, sorted(edges), rng.randint(1, 4))
    finally:
        s.close()


def _check_reachability(
    s: LineageStore, ids: list[str], edges: list[tuple[str, str]], root_count: int
) -> None:
    s.add_nodes([_node(i, seq=k) for k, i in enumerate(ids)])
    s.add_edges([Edge(p, c, "e") for p, c in edges])
    g = nx.DiGraph()
    g.add_nodes_from(ids)
    g.add_edges_from(edges)
    roots = ids[:root_count]
    expected: set[str] = set()
    for r in roots:
        expected |= {r} | nx.descendants(g, r)
    assert s.reachable(roots) == expected
