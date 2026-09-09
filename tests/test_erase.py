"""Phase 3: suppression before reclaim (3.2), reclaim + physical probe (3.3), saga/DLQ (3.4),
CLI exit codes and refusals (3.5). Runs against every backend available on this machine."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _pipeline, _stores
from tests.conftest import requires_langchain
from tombstone.commands.erase import run_erase
from tombstone.commands.trace import run_trace
from tombstone.erase.journal import Journal
from tombstone.errors import ConfirmationRequired, LineageGapError, SagaError
from tombstone.lineage.stamp import K_EMBED
from tombstone.model.artifacts import ArtifactKind, Scope
from tombstone.model.status import VerifyLevel
from tombstone.receipt.ledger import Ledger
from tombstone.stores.base import ProbeSet
from tombstone.verify.logical import build_probe_set

pytestmark = requires_langchain

ALL = ["chroma", "faiss", "qdrant", "pgvector"]


def _backends(request: pytest.FixtureRequest) -> tuple[list[str], str | None]:
    avail = []
    dsn = None
    for b in ALL:
        try:
            _stores.skip_unless(b)
        except pytest.skip.Exception:
            continue
        if b == "pgvector":
            try:
                dsn = request.getfixturevalue("pg_database")
            except pytest.skip.Exception:
                continue
        avail.append(b)
    if not avail:
        pytest.skip("no vector backend available")
    return avail, dsn


def _probe_queries(h: dict, subject: str) -> list[str]:
    corpus = h["corpus"]
    docs = [d for d in corpus if d.subject == subject]
    qs = []
    for d in docs:
        qs.append(d.text[:80])
        if d.canary:
            qs.append(d.canary.token)
            qs.append(d.canary.sentence)
    return qs


# --- 3.2 suppression alone hides everything, reclaim disabled ------------------------------------


def test_suppression_hides_on_every_query_path_before_reclaim(
    tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    backends, dsn = _backends(request)
    h = _pipeline.build(tmp_path, backends, dsn)
    rt = h["rt"]
    t, _ = run_trace(rt, "S-0003")
    embeds = [a for a in t.artifacts if a.kind is ArtifactKind.EMBED]
    assert embeds
    # power check: before erasure, the subject's chunks ARE retrievable on every path
    hits_before = 0
    for b, vs in h["stores"].items():
        for q in _probe_queries(h, "S-0003")[:4]:
            hits_before += sum(
                1
                for d in vs.similarity_search(q, k=5)
                if d.metadata.get(K_EMBED) in {a.artifact_id for a in embeds}
            )
    assert hits_before > 0, "probes have no power: the subject's chunks were never retrievable"
    code, text, data = run_erase(rt, t.trace_id, "dsr-test-suppress", confirm=True, reclaim=False)
    assert "phase 1 suppress" in text
    targets = {a.artifact_id for a in embeds}
    for b, vs in h["stores"].items():
        backend = vs.backend
        for a in embeds:
            if a.store != backend.name:
                continue
            # id lookup, metadata filter
            assert a.store_key not in backend.get([a.store_key]), (b, "id")
            assert not backend.filter(K_EMBED, a.artifact_id), (b, "filter")
            assert not vs.get_by_ids([a.store_key]), (b, "get_by_ids")
        vec = vs.embedder.embed
        for q in _probe_queries(h, "S-0003"):
            for docs in (
                vs.similarity_search(q, k=20),
                vs.similarity_search_by_vector(vec([q])[0], k=20),
                [d for d, _ in vs.similarity_search_with_score(q, k=20)],
                vs.max_marginal_relevance_search(q, k=20, fetch_k=40),
            ):
                found = {d.metadata.get(K_EMBED) for d in docs} & targets
                assert not found, (b, q[:30], found)
            raw = vs.raw_query(vec([q])[0], k=20)
            assert not ({hh.metadata.get(K_EMBED) for hh in raw} & targets), (b, "raw")
            retr = vs.as_retriever(search_kwargs={"k": 20})
            assert not ({d.metadata.get(K_EMBED) for d in retr.invoke(q)} & targets), (
                b,
                "retriever",
            )
        # the store-level probe agrees: not found, on 20 probe queries per artifact
        for a in embeds:
            if a.store != backend.name:
                continue
            ps = build_probe_set(rt.lineage, a, backend.dims, k=40)
            assert not backend.probe_logical(a, ps).found
    # caches: entries derived from S-0003 chunks are gone
    for a in t.artifacts:
        if a.kind is ArtifactKind.CACHE:
            store = rt.store(a.store)
            assert not store.probe_logical(a, ProbeSet(artifact_id=a.artifact_id)).found
    # but the bytes are still there (reclaim was disabled): receipt says UNVERIFIED/RESIDUAL, not VERIFIED
    counts = {k: v for k, v in data["counts"].items()}
    assert counts.get("verified", 0) == 0 or code == 2
    assert code == 2
    for s in data["statuses"]:
        assert s["outcome"] != "verified" or s["artifact"]["kind"] in {"cache"}, s
    # other subjects unaffected
    other_hits = 0
    for vs in h["stores"].values():
        for d in vs.similarity_search(_probe_queries(h, "S-0001")[0], k=5):
            other_hits += 1
    assert other_hits > 0
    rt.close()


# --- 3.3 reclaim then physical probe, idempotent -------------------------------------------------


@pytest.mark.parametrize("backend", ALL)
def test_reclaim_removes_bytes_and_is_idempotent(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    _stores.skip_unless(backend)
    dsn = request.getfixturevalue("pg_database") if backend == "pgvector" else None
    store = _stores.make_backend(backend, tmp_path, pg_dsn=dsn)
    if VerifyLevel.PHYSICAL not in store.capabilities:
        pytest.skip(
            f"{backend}: no physical capability here ({store.physical_unsupported_reason()})"
        )
    emb = _stores.embedder()
    docs = _stores.stamped_docs(request.getfixturevalue("pepper"), n_subjects=3, per_subject=1)[:12]
    with _stores.lineage_and_capture(tmp_path) as (lineage, capture):
        texts = [x for x, _ in docs]
        recs = capture.prepare_embeds(
            store.name,
            emb.name,
            [f"k{i}" for i in range(12)],
            emb.embed(texts),
            [m for _, m in docs],
            texts,
        )
        store.add(recs)
        target = recs[0].embed_node.ref()
        others = [r.embed_node.ref() for r in recs[1:]]
        # power: the vector bytes are present before anything happens
        assert store.probe_physical(target).found, "physical probe has no power on this backend"
        # native delete leaves bytes (the Ghost Vectors finding) — record it either way
        store.native_delete([target.store_key])
        after_native = store.probe_physical(target)
        r = store.reclaim([target])
        assert not r.noop and r.method
        assert not store.probe_physical(target).found, store.probe_physical(target)
        # survivors intact and retrievable
        for o in others:
            assert store.probe_physical(o).found
            assert o.store_key in store.get([o.store_key])
        top = store.query(emb.embed([texts[5]])[0], 3)
        assert top and top[0].key == "k5"
        # idempotent: a second reclaim is a no-op
        r2 = store.reclaim([target])
        assert r2.noop
        assert store.count() == 11
        print(f"{backend}: native delete left bytes = {after_native.found}; reclaim = {r.method}")
    store.close()


# --- 3.5 CLI: refusals and exit codes --------------------------------------------------------------


def test_erase_refusals_and_exit_codes(
    tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _stores.skip_unless("faiss")
    h = _pipeline.build(tmp_path, ["faiss"])
    rt = h["rt"]
    t, _ = run_trace(rt, "S-0003")
    with pytest.raises(ConfirmationRequired):
        run_erase(rt, t.trace_id, "dsr-1", confirm=False)
    with pytest.raises(SagaError, match="trace id"):
        run_erase(rt, "01NOPE", "dsr-1", confirm=True)
    assert Journal(rt.inst.journal_path).records() == []  # nothing happened
    # a gap → refusal without --accept-gaps; with it → UNVERIFIED(lineage-gap) for that store
    emb = _stores.embedder()
    legacy = [f"legacy {i}" for i in range(5)]
    from tombstone.lineage.capture import EmbedRecord

    h["stores"]["faiss"].backend._add(
        [
            EmbedRecord(f"z{i}", emb.embed([x])[0], {"legacy": True}, x, None, None)
            for i, x in enumerate(legacy)
        ]
    )  # type: ignore[arg-type]
    t2, _ = run_trace(rt, "S-0003")
    assert t2.gaps
    with pytest.raises(LineageGapError, match="accept-gaps"):
        run_erase(rt, t2.trace_id, "dsr-2", confirm=True)
    code, text, data = run_erase(rt, t2.trace_id, "dsr-2", confirm=True, accept_gaps=True)
    assert code == 2
    faiss_statuses = [s for s in data["statuses"] if s["artifact"]["store"] == "faiss:kb-v1"]
    assert faiss_statuses and all(
        s["outcome"] == "unverified" and s["rule_id"] == "lineage_gap" for s in faiss_statuses
    )
    assert "UNVERIFIED-lineage-gap" in text
    # a second erase of the same trace is refused (already has a receipt)
    with pytest.raises(SagaError, match="already erased"):
        run_erase(rt, t2.trace_id, "dsr-3", confirm=True, accept_gaps=True)
    ledger = Ledger(rt.inst.ledger_path)
    assert ledger.verify() >= 1 and len(ledger.receipts()) == 1
    rt.close()


def test_full_erase_all_verified_exit_zero(
    tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Demo 2 shape: every self-hosted store VERIFIED(physical) → exit 0 (adapter excluded until Phase 4)."""
    monkeypatch.chdir(tmp_path)
    backends, dsn = _backends(request)
    h = _pipeline.build(tmp_path, backends, dsn, rag_subjects=("S-0001",))
    rt = h["rt"]
    t, _ = run_trace(rt, "S-0001")
    assert not t.gaps and not t.third_party_hits
    code, text, data = run_erase(rt, t.trace_id, "dsr-2026-0912", confirm=True)
    print(text)
    bad = [s for s in data["statuses"] if s["outcome"] != "verified"]
    assert not bad, json.dumps(bad, indent=1)[:2000]
    assert code == 0
    assert "OUT_OF_SCOPE 3" in text and "ed25519 signed" in text
    assert data["counts"]["verified"] == len(t.artifacts)
    # every artifact physically verified (docstore, vectors, caches, dataset)
    assert all(s["level"] == "physical" for s in data["statuses"])
    # the subject is gone from every store and other subjects survive
    for vs in h["stores"].values():
        assert not vs.similarity_search(_probe_queries(h, "S-0001")[0], k=5) or all(
            d.metadata.get("tombstone.subject") != t.subject.hmac
            for d in vs.similarity_search(_probe_queries(h, "S-0001")[0], k=5)
        )
        assert vs.similarity_search(_probe_queries(h, "S-0002")[0], k=3)
    # journal + ledger chains verify
    assert Journal(rt.inst.journal_path).verify() > 0
    assert Ledger(rt.inst.ledger_path).verify() == 1
    rt.close()


# --- 3.4 chaos: SIGKILL at 15 random points, resume, identical receipt -----------------------------


class _AlwaysFailingStore:
    """Registered in place of a real store to exercise the DLQ."""


def _run_cli(
    args: list[str], cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    e = {**os.environ, "TOMBSTONE_LOG_LEVEL": "error", **(env or {})}
    return subprocess.run(
        [sys.executable, "-m", "tombstone", *args],
        cwd=cwd,
        env=e,
        capture_output=True,
        text=True,
    )


@pytest.mark.slow
def test_chaos_sigkill_resume_produces_identical_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import random

    monkeypatch.chdir(tmp_path)
    _stores.skip_unless("faiss")
    _stores.skip_unless("chroma")
    h = _pipeline.build(tmp_path, ["faiss", "chroma"], rag_subjects=("S-0001",))
    rt = h["rt"]
    t, _ = run_trace(rt, "S-0001")
    assert len(t.artifacts) >= 40, len(t.artifacts)
    rt.close()
    # reference run: a clean erase in a subprocess
    import shutil

    snapshot = tmp_path / "_snapshot"
    shutil.copytree(tmp_path, snapshot, ignore=shutil.ignore_patterns("_snapshot"))

    def fresh_copy(name: str) -> Path:
        dst = tmp_path / name
        shutil.copytree(snapshot, dst)
        # config paths are absolute → rewrite them for the copy
        c = (dst / "tombstone.yaml").read_text().replace(str(tmp_path), str(dst))
        (dst / "tombstone.yaml").write_text(c)
        return dst

    ref_dir = fresh_copy("_ref")
    ref_cfg = str(ref_dir / "tombstone.yaml")
    p = _run_cli(
        [
            "erase",
            "--config",
            ref_cfg,
            "--trace",
            t.trace_id,
            "--reason",
            "dsr-chaos",
            "--confirm",
            "--json",
        ],
        ref_dir,
    )
    if p.returncode not in (0, 2):
        # diagnostics for a pin mismatch: what does the copied manifest look like, and the pins?
        import sqlite3

        m = json.loads((ref_dir / "train" / "manifest.json").read_text())
        pins = (
            sqlite3.connect(str(ref_dir / ".tombstone" / "lineage.db"))
            .execute("select name, reason, payload from pins order by pinned_seq")
            .fetchall()
        )
        raise AssertionError(
            f"reference erase failed rc={p.returncode}\nstderr: {p.stderr[-1500:]}\n"
            f"manifest hash {m['manifest_hash'][:12]} suppressed={sum(1 for e in m['examples'] if e.get('suppressed'))}/{len(m['examples'])}\n"
            f"pins: {[(n, r, json.loads(pl).get('manifest_hash', '')[:12]) for n, r, pl in pins]}"
        )
    ref = json.loads(p.stdout)
    ref_table = sorted(
        (s["artifact"]["artifact_id"], s["outcome"], s["level"], s["rule_id"])
        for s in ref["statuses"]
    )
    n_journal = len(Journal(ref_dir / ".tombstone" / "journal.jsonl").records())
    rng = random.Random(20260908)
    kill_points = sorted(rng.sample(range(2, n_journal - 1), 15))
    for i, kp in enumerate(kill_points):
        d = fresh_copy(f"_chaos{i}")
        c = str(d / "tombstone.yaml")
        killed = _run_cli(
            [
                "erase",
                "--config",
                c,
                "--trace",
                t.trace_id,
                "--reason",
                "dsr-chaos",
                "--confirm",
                "--json",
            ],
            d,
            {"TOMBSTONE_CHAOS_KILL_AFTER": str(kp)},
        )
        assert killed.returncode != 0, "chaos kill did not fire"
        assert not (d / ".tombstone" / "ledger.jsonl").read_text().strip(), (
            "a receipt was written mid-saga"
        )
        j = Journal(d / ".tombstone" / "journal.jsonl")
        assert j.verify() == kp  # journal intact up to the kill
        assert j.open_sagas(), "resumable journal expected"
        resumed = _run_cli(
            [
                "erase",
                "--config",
                c,
                "--trace",
                t.trace_id,
                "--reason",
                "dsr-chaos",
                "--confirm",
                "--json",
            ],
            d,
        )
        assert resumed.returncode in (0, 2), resumed.stderr[-2000:]
        out = json.loads(resumed.stdout)
        table = sorted(
            (s["artifact"]["artifact_id"], s["outcome"], s["level"], s["rule_id"])
            for s in out["statuses"]
        )
        assert table == ref_table, f"kill point {kp}: receipt differs"
        j = Journal(d / ".tombstone" / "journal.jsonl")
        assert j.verify() > kp and not j.open_sagas()
        # no duplicated reclaim side effects: each store's reclaim did real work at most once
        ends = [
            r
            for r in j.records()
            if r.type == Journal.STEP_END
            and r.body["step_id"].startswith("reclaim:")
            and r.body["ok"]
            and not r.body["noop"]
        ]
        per_store: dict[str, int] = {}
        for r in ends:
            per_store[r.body["step_id"]] = per_store.get(r.body["step_id"], 0) + 1
        assert all(v == 1 for v in per_store.values()), per_store
        assert Ledger(d / ".tombstone" / "ledger.jsonl").verify() == 1
        shutil.rmtree(d)


def test_dlq_captures_failing_store_and_retry_drains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _stores.skip_unless("faiss")
    h = _pipeline.build(tmp_path, ["faiss"], rag_subjects=("S-0002",))
    rt = h["rt"]
    t, _ = run_trace(rt, "S-0002")
    # make the FAISS store always fail at reclaim
    backend = rt.store("faiss:kb-v1")
    original = backend.reclaim

    def boom(refs):  # noqa: ANN001, ANN202
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(backend, "reclaim", boom)
    code, text, data = run_erase(rt, t.trace_id, "dsr-dlq", confirm=True)
    assert code == 2 and "dead-letter queue" in text
    dlq = [r for r in Journal(rt.inst.journal_path).records() if r.type == Journal.DLQ]
    assert dlq and dlq[0].body["store"] == "faiss:kb-v1" and "disk on fire" in dlq[0].body["error"]
    faiss_statuses = [s for s in data["statuses"] if s["artifact"]["store"] == "faiss:kb-v1"]
    assert all(s["outcome"] == "unverified" and s["rule_id"] == "dlq" for s in faiss_statuses)
    # other stores were still processed and verified
    assert any(
        s["outcome"] == "verified"
        for s in data["statuses"]
        if s["artifact"]["store"] != "faiss:kb-v1"
    )
    # the receipt was written only after every artifact had a terminal status
    assert len(Ledger(rt.inst.ledger_path).receipts()) == 1
    # retry: store recovered → new receipt, everything verified
    monkeypatch.setattr(backend, "reclaim", original)
    code2, text2, data2 = run_erase(rt, t.trace_id, "dsr-dlq-retry", confirm=True, retry=True)
    assert code2 == 0, text2
    assert len(Ledger(rt.inst.ledger_path).receipts()) == 2
    assert (
        Ledger(rt.inst.ledger_path).receipts()[1].prev_receipt_hash
        == Ledger(rt.inst.ledger_path).records()[0].hash
    )
    rt.close()


# --- 3.1 managed instance: a role without maintenance rights -------------------------------------


@pytest.mark.pg
def test_managed_pg_role_reports_logical_only(pg_database: str, pg) -> None:  # noqa: ANN001
    import psycopg

    from tombstone.errors import NotSupported
    from tombstone.stores.pgvector import PgVectorStore

    owner = PgVectorStore(
        "pgvector:kb-v1", pg_database, "documents", embedding_model="hash-embed-64", dims=64
    )
    assert VerifyLevel.PHYSICAL in owner.capabilities
    with psycopg.connect(pg_database, autocommit=True) as conn:
        conn.execute("DROP ROLE IF EXISTS limited")
        conn.execute("CREATE ROLE limited LOGIN PASSWORD 'limited'")
        conn.execute("GRANT USAGE ON SCHEMA public TO limited")
        conn.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON documents TO limited")
    base, _, dbname = pg_database.rpartition("/")
    host = base.split("@")[-1]
    limited_dsn = f"postgresql://limited:limited@{host}/{dbname}"
    managed = PgVectorStore(
        "pgvector:kb-v1", limited_dsn, "documents", embedding_model="hash-embed-64", dims=64
    )
    assert managed.capabilities == frozenset({VerifyLevel.LOGICAL})
    assert managed.maintenance == "none" and not managed.can_read_files
    from tombstone.model.artifacts import ArtifactRef

    ref = ArtifactRef(
        "01X", ArtifactKind.EMBED, "pgvector:kb-v1", "k0", Scope("default"), "h" * 64, "00" * 128
    )
    with pytest.raises(NotSupported) as ei:
        managed.probe_physical(ref)
    msg = str(ei.value)
    assert "pg_read_binary_file" in msg and "owner" in msg
    r = managed.reclaim([ref])
    assert "not permitted" in r.method and "owner role" in r.detail
    managed.close()
    owner.close()
