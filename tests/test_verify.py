"""5.1/5.2 probe power, Demo 1 audit (native delete), 5.5 replay, 5.6 independent verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import _pipeline, _stores
from tests.conftest import requires_langchain
from tombstone.commands.erase import run_erase
from tombstone.commands.trace import run_trace
from tombstone.errors import ReplayMismatch
from tombstone.lineage.stamp import K_EMBED
from tombstone.model.artifacts import ArtifactKind
from tombstone.receipt.ledger import Ledger
from tombstone.receipt.replay import assert_replay, replay_ledger
from tombstone.verify.audit import audit_trace, render_audit_report
from tombstone.verify.independent import verify_receipt_independently
from tombstone.verify.logical import build_probe_set, derived_queries

pytestmark = requires_langchain


def test_derived_queries() -> None:
    assert derived_queries("") == []
    qs = derived_queries("one two three four five six seven eight nine ten eleven twelve")
    assert len(qs) == 3 and qs[0].startswith("one two")


@pytest.mark.parametrize("backend", ["chroma", "faiss", "qdrant", "pgvector"])
def test_probes_have_power_then_pass_after_delete_and_suppress(
    backend: str, tmp_path: Path, pepper: bytes, request: pytest.FixtureRequest
) -> None:
    """5.1: a present artifact fails the logical probe; a deleted one passes; a suppressed-not-reclaimed
    one passes (Hard Rule 5). 5.2: the physical probe finds a soft-deleted vector; not after reclaim."""
    _stores.skip_unless(backend)
    dsn = request.getfixturevalue("pg_database") if backend == "pgvector" else None
    store = _stores.make_backend(backend, tmp_path, pg_dsn=dsn)
    emb = _stores.embedder()
    docs = _stores.stamped_docs(pepper, n_subjects=3, per_subject=1)[:12]
    with _stores.lineage_and_capture(tmp_path) as (lineage, capture):
        texts = [x for x, _ in docs]
        recs = capture.prepare_embeds(
            store.name,
            emb.name,
            [f"k{i}" for i in range(12)],
            emb.embed(texts),
            [m for _, m in docs],
            texts,
            emb.embed,
        )
        store.add(recs)
        a = recs[0].embed_node.ref()
        b = recs[1].embed_node.ref()
        ps_a = build_probe_set(lineage, a, store.dims, 40)
        assert len(ps_a.vectors) >= 2  # padded fingerprint + probe-table queries
        assert store.probe_logical(a, ps_a).found  # power
        if "physical" in {c.value for c in store.capabilities}:
            assert store.probe_physical(a).found
        # native delete → logical passes, physical (Ghost Vectors) still finds bytes on most stores
        store.native_delete([a.store_key])
        assert not store.probe_logical(a, ps_a).found
        if "physical" in {c.value for c in store.capabilities}:
            res = store.probe_physical(a)
            print(f"{backend}: physical residue after native delete: {res.found} {res.locations}")
        # suppress-not-reclaim → logical passes, bytes present
        store.suppress([b])
        assert not store.probe_logical(b, build_probe_set(lineage, b, store.dims, 40)).found
        if "physical" in {c.value for c in store.capabilities}:
            assert store.probe_physical(b).found
            store.reclaim([a, b])
            assert not store.probe_physical(a).found and not store.probe_physical(b).found
    store.close()


def test_demo1_audit_after_native_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator deletes the way everyone deletes; verify shows what that reached."""
    monkeypatch.chdir(tmp_path)
    _stores.skip_unless("chroma")
    _stores.skip_unless("faiss")
    h = _pipeline.build(tmp_path, ["chroma", "faiss"], rag_subjects=("S-0004",))
    rt = h["rt"]
    t, _ = run_trace(rt, "S-0004")
    embeds = [a for a in t.artifacts if a.kind is ArtifactKind.EMBED]
    # native delete on both vector stores + drop the source rows
    for vs in h["stores"].values():
        keys = [a.store_key for a in embeds if a.store == vs.backend.name]
        vs.delete(keys)
    docs = rt.store("docs")
    docs.native_delete(
        [a.store_key for a in t.artifacts if a.kind in {ArtifactKind.SOURCE, ArtifactKind.CHUNK}]
    )
    t2, _ = run_trace(rt, "S-0004")
    report = audit_trace(rt, t2)
    text = render_audit_report(report)
    print(text)
    states = {(r["kind"], r["store"]): r["state"] for r in report.to_dict()["rows"]}
    # source/chunk rows: logically gone, but the sqlite file still holds the bytes → HIDDEN
    assert all(v in {"HIDDEN", "GONE"} for k, v in states.items() if k[0] in {"source", "chunk"})
    embed_states = {v for k, v in states.items() if k[0] == "embed"}
    assert "PRESENT" not in embed_states  # native delete is logically effective
    assert "HIDDEN" in embed_states  # ...and physically ineffective on at least one store
    assert any(v == "PRESENT" for k, v in states.items() if k[0] == "cache")  # caches untouched
    assert any(v == "PRESENT" for k, v in states.items() if k[0] == "train")
    assert report.recoverable >= 3 and "NOT ERASED" in text
    assert "verdict:" in text and "recoverable" in text
    rt.close()


def test_replay_fifty_receipts_and_lattice_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _stores.skip_unless("faiss")
    from tests import _corpus

    h = _pipeline.build(tmp_path, ["faiss"], rag_subjects=())
    rt = h["rt"]
    subjects = _corpus.subjects()
    # many receipts: erase each subject twice through retries is cheap; generate 50 via
    # single-artifact traces would need more subjects, so use retries on 8 subjects
    n = 0
    for subj in subjects:
        t, _ = run_trace(rt, subj, with_store_gaps=False)
        run_erase(rt, t.trace_id, f"dsr-{subj}", confirm=True)
        n += 1
        for k in range(6):
            run_erase(rt, t.trace_id, f"dsr-{subj}-retry{k}", confirm=True, retry=True)
            n += 1
        if n >= 50:
            break
    ledger = Ledger(rt.inst.ledger_path)
    assert len(ledger.receipts()) >= 50
    report = replay_ledger(rt.inst.ledger_path, rt.inst.journal_path)
    assert report.ok and report.matched == report.receipts >= 50
    assert_replay(rt.inst.ledger_path, rt.inst.journal_path)
    # change a lattice constant: rule 5 now treats physical as supported → VERIFIED where it was UNVERIFIED... we
    # flip 'verified' to require model_applicable, which changes recorded rule ids
    import tombstone.verify.levels as levels

    original = levels.assign

    def broken(f):  # noqa: ANN001, ANN202
        s = original(f)
        if s.rule_id == "verified":
            from tombstone.model.status import ArtifactStatus, Outcome

            return ArtifactStatus(
                s.artifact,
                s.suppressed_at,
                s.reclaimed,
                Outcome.UNVERIFIED,
                s.level,
                "lattice changed",
                s.measurement,
                "changed",
            )
        return s

    monkeypatch.setattr("tombstone.receipt.replay.assign", broken)
    with pytest.raises(ReplayMismatch) as ei:
        assert_replay(rt.inst.ledger_path, rt.inst.journal_path)
    first = ledger.receipts()[0].receipt_id
    assert first in str(ei.value)
    # semantics version bump is reported
    monkeypatch.setattr("tombstone.receipt.replay.assign", original)
    rep = replay_ledger(rt.inst.ledger_path, rt.inst.journal_path, lattice_version=2)
    assert (
        rep.version_mismatches
        and "v1" in rep.version_mismatches[0]
        and "v2" in rep.version_mismatches[0]
    )
    rt.close()


def test_independent_verifier_catches_reinserted_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _stores.skip_unless("faiss")
    h = _pipeline.build(tmp_path, ["faiss"], rag_subjects=("S-0002",))
    rt = h["rt"]
    t, _ = run_trace(rt, "S-0002")
    code, text, data = run_erase(rt, t.trace_id, "dsr-iv", confirm=True)
    # Exit 2 is correct here when the fixture's shared boilerplate leaves artifacts whose
    # bytes other live artifacts also hold: those report UNVERIFIED(duplicate content),
    # because no byte scan can attribute a shared byte. This fixture is unusually
    # repetitive (56 unique chunks out of 88) so it hits that case hard; the measured
    # corpus sits near 8%.
    assert code in (0, 2), text
    assert all(
        s["outcome"] == "verified" or "duplicate content" in s.get("reason", "")
        for s in data["statuses"]
    ), text
    path = rt.inst.receipts_dir / f"{data['receipt_id']}.json"
    pub = rt.inst.public_key_path
    cfg = h["cfg_path"]
    rt.close()
    res = verify_receipt_independently(path, pub, tmp_path / ".tombstone" / "ledger.jsonl", cfg)
    assert res["ok"], res["text"]
    # the operator's own public key is passed here, so this is a statement about origin
    assert "signature: ed25519 OK" in res["text"] and "chain: OK" in res["text"]
    assert str(pub) in res["text"], "the verifier must name the key it trusted"
    # ...and without it, the same receipt is only self-consistent, which it must say plainly
    solo = verify_receipt_independently(path, None, tmp_path / ".tombstone" / "ledger.jsonl", cfg)
    assert "signature: ed25519 self-asserted" in solo["text"], solo["text"]
    # re-insert one deleted vector behind the tool's back
    from tombstone.registry import Runtime

    rt2 = Runtime.load(cfg)
    backend = rt2.store("faiss:kb-v1")
    # A VERIFIED one: an artifact reported UNVERIFIED(duplicate content) is one whose bytes a
    # scan cannot attribute, so re-inserting it is invisible by design and the receipt still
    # agrees. Statuses are ordered by ULID, whose random tail varies between runs, so the
    # first faiss row is sometimes such an artifact — that was a flaky pick, not a bug.
    victim = next(
        s
        for s in data["statuses"]
        if s["artifact"]["store"] == "faiss:kb-v1" and s["outcome"] == "verified"
    )
    emb = _stores.embedder()
    from tombstone.lineage.capture import EmbedRecord

    text_of = next(d.text for d in h["corpus"] if d.subject == "S-0002")
    backend._add(
        [
            EmbedRecord(
                victim["artifact"]["store_key"],
                emb.embed([text_of])[0],
                {K_EMBED: victim["artifact"]["artifact_id"], "tombstone.suppressed": False},
                text_of,
                None,
                None,
            )
        ]
    )  # type: ignore[arg-type]
    rt2.close()
    res2 = verify_receipt_independently(path, pub, tmp_path / ".tombstone" / "ledger.jsonl", cfg)
    assert not res2["ok"]
    assert any(
        r["artifact_id"] == victim["artifact"]["artifact_id"] and not r["agree"]
        for r in res2["rows"]
    )
    assert "RESIDUAL" in res2["text"] and "NOT reproduced" in res2["text"]
    # tampered receipt fails the signature
    blob = json.loads(path.read_text())
    blob["reason"] = "forged"
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(blob))
    res3 = verify_receipt_independently(forged, pub, None, cfg)
    assert not res3["ok"] and "INVALID" in res3["text"]


@pytest.mark.parametrize("backend", ["chroma", "faiss", "qdrant", "pgvector"])
def test_survivor_with_identical_content_is_not_the_deleted_record_s_residue(
    backend: str, tmp_path: Path, pepper: bytes, request: pytest.FixtureRequest
) -> None:
    """6.3: two subjects hold byte-identical text, so their vectors share a fingerprint. Erase and
    reclaim one; the survivor's bytes are still on disk and the byte-scan still matches them.

    Those bytes are the survivor's. The shipped verifier says so, and the residue benchmark must
    reach the same verdict on the same bytes — the bench used to call this residue because the
    deleted record's pattern count "did not drop below its baseline", which is what a rebuild does
    to a surviving duplicate, and the published matrix then disagreed with ``tombstone verify``.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
    from residue.run_residue import physical_after, physical_baseline
    from tombstone.verify.independent import _physical_observation

    _stores.skip_unless(backend)
    dsn = request.getfixturevalue("pg_database") if backend == "pgvector" else None
    store = _stores.make_backend(backend, tmp_path, pg_dsn=dsn)
    if "physical" not in {c.value for c in store.capabilities}:
        pytest.skip(f"{backend} has no physical level here")
    emb = _stores.embedder()
    docs = _stores.stamped_docs(pepper, n_subjects=2, per_subject=1)[:2]
    shared = "The agent noted that a satisfaction survey was sent after the call."
    texts = [shared, shared]  # identical text => identical vector => identical fingerprint
    with _stores.lineage_and_capture(tmp_path) as (_lineage, capture):
        recs = capture.prepare_embeds(
            store.name,
            emb.name,
            ["dup-a", "dup-b"],
            emb.embed(texts),
            [m for _, m in docs],
            texts,
            emb.embed,
        )
        store.add(recs)
        a = recs[0].embed_node.ref()
        b = recs[1].embed_node.ref()
        assert a.embedding_fingerprint == b.embedding_fingerprint, (
            "fixture must share a fingerprint"
        )
        base = physical_baseline(store, [a])
        store.reclaim([a])  # a is gone; b survives with the same bytes

        assert store.live_content_duplicates(a) > 0, "the survivor must still hold those bytes"
        present, why = _physical_observation(store, a)
        assert present is False and "live record" in why, why
        pa = physical_after(store, [a], base)
        assert pa is not None
        assert pa["physical_residue"] == 0.0, "bench must agree with the verifier"
        assert pa["attributed_to_live_duplicate"] == 1.0
        assert pa["probe_power"] == 1.0, "the probe must have been able to see it before"
        # and the survivor itself is untouched
        assert store.get(["dup-b"]).get("dup-b") is not None
    store.close()


@pytest.mark.parametrize("backend", ["chroma", "faiss", "qdrant", "pgvector"])
def test_batch_probe_sees_the_same_bytes_as_the_single_probe(
    backend: str, tmp_path: Path, pepper: bytes, request: pytest.FixtureRequest
) -> None:
    """The batched physical probe and the single-ref one must agree, artifact for artifact.

    pgvector reads its files back through ``pg_read_binary_file`` and so reports no
    ``persisted_files()`` of its own; it inherited a batch probe that walks exactly that empty
    list, and every batched probe silently found nothing. The residue benchmark probes in batches.
    """
    _stores.skip_unless(backend)
    dsn = request.getfixturevalue("pg_database") if backend == "pgvector" else None
    store = _stores.make_backend(backend, tmp_path, pg_dsn=dsn)
    if "physical" not in {c.value for c in store.capabilities}:
        pytest.skip(f"{backend} has no physical level here")
    emb = _stores.embedder()
    docs = _stores.stamped_docs(pepper, n_subjects=3, per_subject=1)[:6]
    with _stores.lineage_and_capture(tmp_path) as (_lineage, _capture0):
        texts = [x for x, _ in docs]
        recs = _capture0.prepare_embeds(
            store.name,
            emb.name,
            [f"bk{i}" for i in range(len(docs))],
            emb.embed(texts),
            [m for _, m in docs],
            texts,
            emb.embed,
        )
        store.add(recs)
        refs = [r.embed_node.ref() for r in recs]
        batch = store.probe_physical_batch(refs)
        assert any(v.found for v in batch.values()), "batch probe found nothing while present"
        for a in refs:
            single = store.probe_physical(a)
            b = batch[a.artifact_id]
            assert b.found == single.found, (
                f"{a.artifact_id}: batch {b.found} != single {single.found}"
            )
            for k in ("matches_artifact_id", "matches_f32le"):
                if k in single.measurement:
                    assert b.measurement.get(k) == single.measurement[k], k
    store.close()
