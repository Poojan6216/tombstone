"""8.3: edge cases. Nothing corrupts the journal or ledger, nothing writes a receipt for an
incomplete saga, every failure names its cause."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests import _stores
from tests.conftest import requires_faiss, requires_langchain
from tombstone.commands.erase import run_erase
from tombstone.commands.trace import run_trace
from tombstone.erase.journal import Journal
from tombstone.errors import LineageGapError, LockTimeout, PinMismatch, SagaError
from tombstone.receipt.ledger import Ledger

pytestmark = [requires_langchain, requires_faiss]


def _app(tmp_path: Path, subjects: tuple[str, ...] = ()) -> dict:
    from tests import _pipeline

    return _pipeline.build(tmp_path, ["faiss"], rag_subjects=subjects)


def test_empty_subject_and_unknown_subject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    h = _app(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        run_trace(h["rt"], "")
    with pytest.raises(LineageGapError, match="no lineage records"):
        run_trace(h["rt"], "S-9999")
    h["rt"].close()


def test_unicode_subject_id_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_core.documents import Document

    from tombstone.integrations.langchain import TombstoneVectorStore
    from tombstone.lineage.stamp import stamp

    monkeypatch.chdir(tmp_path)
    h = _app(tmp_path)
    rt = h["rt"]
    vs = TombstoneVectorStore.from_config("faiss:kb-v1", config=h["cfg_path"])
    subj = "Zoë Ångström-Łukasiewicz 李雷"
    doc = stamp(
        Document(
            page_content="unicode subject document about a refund", metadata={"source": "u.txt"}
        ),
        subj,
        "u.txt",
        "default",
        pepper=rt.pepper(),
    )
    vs.add_documents([doc])
    t, _ = run_trace(rt, subj, with_store_gaps=False)
    assert t.artifacts
    code, _text, _data = run_erase(rt, t.trace_id, "dsr-unicode", confirm=True)
    assert code == 0
    for p in (rt.inst.root).rglob("*"):
        if p.is_file():
            assert subj.encode() not in p.read_bytes()
    rt.close()


def test_two_megabyte_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_core.documents import Document

    from tombstone.integrations.langchain import TombstoneVectorStore
    from tombstone.lineage.stamp import stamp

    monkeypatch.chdir(tmp_path)
    h = _app(tmp_path)
    rt = h["rt"]
    vs = TombstoneVectorStore.from_config("faiss:kb-v1", config=h["cfg_path"])
    big = ("large document text " * 100_000)[: 2 * 1024 * 1024]
    doc = stamp(
        Document(page_content=big, metadata={"source": "big.txt"}),
        "S-BIG",
        "big.txt",
        "default",
        pepper=rt.pepper(),
    )
    vs.add_documents([doc])
    t, _ = run_trace(rt, "S-BIG", with_store_gaps=False)
    code, _, _ = run_erase(rt, t.trace_id, "dsr-big", confirm=True)
    assert code == 0
    rt.close()


def test_store_returning_success_without_deleting_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    h = _app(tmp_path)
    rt = h["rt"]
    backend = rt.store("faiss:kb-v1")
    from tombstone.stores.base import ReclaimResult

    monkeypatch.setattr(
        backend, "reclaim", lambda refs: ReclaimResult(noop=False, method="lied", measurement={})
    )
    t, _ = run_trace(rt, "S-0001", with_store_gaps=False)
    code, text, data = run_erase(rt, t.trace_id, "dsr-liar", confirm=True)
    faiss_rows = [s for s in data["statuses"] if s["artifact"]["store"] == "faiss:kb-v1"]
    assert code == 2 and all(s["outcome"] == "residual" for s in faiss_rows)
    assert "RESIDUAL" in text
    rt.close()


def test_store_down_mid_reclaim_goes_to_dlq_no_receipt_until_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    h = _app(tmp_path)
    rt = h["rt"]
    backend = rt.store("faiss:kb-v1")

    def down(refs):  # noqa: ANN001, ANN202
        raise ConnectionError("store unreachable")

    monkeypatch.setattr(backend, "reclaim", down)
    t, _ = run_trace(rt, "S-0002", with_store_gaps=False)
    code, _text, data = run_erase(rt, t.trace_id, "dsr-down", confirm=True)
    assert code == 2
    dlq = [r for r in Journal(rt.inst.journal_path).records() if r.type == Journal.DLQ]
    assert dlq and "store unreachable" in dlq[0].body["error"]
    assert all(
        s["reason"].startswith("dlq:")
        for s in data["statuses"]
        if s["artifact"]["store"] == "faiss:kb-v1"
    )
    assert (
        Ledger(rt.inst.ledger_path).verify() == 1
    )  # written only after every artifact is terminal
    rt.close()


def test_duplicate_example_in_manifest_is_rejected(tmp_path: Path, pepper: bytes) -> None:
    from tombstone.train.dataset import DatasetStore, build_dataset

    docs = _stores.stamped_docs(pepper, n_subjects=2, per_subject=1)[:4]
    with _stores.lineage_and_capture(tmp_path) as (_lineage, capture):
        chunks = [(capture.ensure_chunk(md, text)[0], text) for text, md in docs]
        path = tmp_path / "train" / "manifest.json"
        build_dataset(capture, "ft", path, chunks, shards=2)
        blob = json.loads(path.read_text())
        blob["examples"].append(dict(blob["examples"][0]))  # duplicate row
        path.write_text(json.dumps(blob))
        with pytest.raises(PinMismatch):
            DatasetStore("ft", path).manifest()


def test_two_lineage_dbs_pointing_at_one_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second installation over the same index: its trace sees the store's entries as unlineaged."""
    monkeypatch.chdir(tmp_path)
    h = _app(tmp_path)
    h["rt"].close()
    other = tmp_path / "other"
    other.mkdir()
    from tombstone.commands.init import run_init

    run_init(other)
    cfg2 = other / "tombstone.yaml"
    cfg2.write_text(
        cfg2.read_text().replace(
            "stores:\n",
            f'stores:\n  - {{ name: "faiss:kb-v1", kind: faiss, path: {tmp_path}/faiss/kb.index, embedding: hash-embed-64 }}\n',
        )
    )
    from tombstone.registry import Runtime

    rt2 = Runtime.load(cfg2)
    from tombstone.lineage.gaps import detect_gaps

    rep = detect_gaps(rt2.lineage, rt2.scope, rt2.all_stores())
    assert rep.has_gaps and "faiss:kb-v1" in rep.stores_without_capture
    rt2.close()


def test_disk_full_during_journal_write_leaves_chain_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    h = _app(tmp_path)
    rt = h["rt"]
    t, _ = run_trace(rt, "S-0003", with_store_gaps=False)
    import tombstone.chain as chain_mod

    real_open = Path.open
    calls = {"n": 0}

    def flaky_open(self, *a, **k):  # noqa: ANN001, ANN202
        if "a" in (a[0] if a else k.get("mode", "")) and self.name == "journal.jsonl":
            calls["n"] += 1
            if calls["n"] == 3:
                raise OSError(28, "No space left on device")
        return real_open(self, *a, **k)

    monkeypatch.setattr(Path, "open", flaky_open)
    with pytest.raises((OSError, SagaError)):
        run_erase(rt, t.trace_id, "dsr-full", confirm=True)
    monkeypatch.setattr(Path, "open", real_open)
    j = Journal(rt.inst.journal_path)
    assert j.verify() >= 1  # every record that made it to disk is intact
    assert Ledger(rt.inst.ledger_path).receipts() == []  # no receipt for an incomplete saga
    # and the saga resumes cleanly
    code, _text, _data = run_erase(rt, t.trace_id, "dsr-full", confirm=True)
    assert code == 0
    assert chain_mod.HashChain(rt.inst.journal_path).verify() > 0
    rt.close()


def test_receipt_directory_with_five_thousand_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5,000 receipt files + a long ledger: listing and lookup stay correct."""
    monkeypatch.chdir(tmp_path)
    h = _app(tmp_path)
    rt = h["rt"]
    from tombstone.model.artifacts import Scope, SubjectRef
    from tombstone.model.status import Receipt

    ledger = Ledger(rt.inst.ledger_path)
    for i in range(5000):
        r = Receipt(
            f"R{i:05d}",
            "T",
            SubjectRef("s" * 64),
            Scope("default"),
            "r",
            (),
            ("backups",),
            {},
            "j" * 64,
            ledger.prev_receipt_hash(),
            "sig",
            public_key="00",
        )
        (rt.inst.receipts_dir / f"{r.receipt_id}.json").write_text(json.dumps(r.to_dict()))
        if i % 500 == 0:
            ledger.append(r)
    assert len(list(rt.inst.receipts_dir.iterdir())) == 5000
    assert ledger.verify() == 10
    assert ledger.find("R04500") is not None and ledger.find("R00001") is None
    rt.close()


def test_subject_with_ten_thousand_artifacts_traces(tmp_path: Path, pepper: bytes) -> None:
    from tombstone.lineage.trace import trace
    from tombstone.model.artifacts import ArtifactKind, Scope, SubjectRef
    from tombstone.model.lineage import Edge, Node

    with _stores.lineage_and_capture(tmp_path) as (lineage, _capture):
        subj = SubjectRef.from_raw("S-BIG", pepper)
        nodes = [
            Node(
                "S000000",
                ArtifactKind.SOURCE,
                "docs",
                "S000000",
                Scope("default"),
                "h" * 64,
                None,
                subj.hmac,
                0,
            )
        ]
        edges = []
        for i in range(1, 10_001):
            nodes.append(
                Node(
                    f"C{i:06d}",
                    ArtifactKind.CHUNK,
                    "docs",
                    f"C{i:06d}",
                    Scope("default"),
                    "h" * 64,
                    None,
                    subj.hmac,
                    i,
                )
            )
            edges.append(Edge("S000000", f"C{i:06d}", "chunk"))
        lineage.add_nodes(nodes)
        lineage.add_edges(edges)
        import time

        t0 = time.perf_counter()
        t = trace(subj, Scope("default"), lineage.snapshot(Scope("default")))
        assert len(t.artifacts) == 10_001
        assert time.perf_counter() - t0 < 30


def test_lock_contention_is_a_clean_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    monkeypatch.chdir(tmp_path)
    h = _app(tmp_path)
    rt = h["rt"]
    rt.cfg.erase.lock_timeout_s = 0.3
    t, _ = run_trace(rt, "S-0004", with_store_gaps=False)
    lock = rt.inst.journal_path.with_suffix(".jsonl.lock")
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        with pytest.raises(LockTimeout):
            run_erase(rt, t.trace_id, "dsr-lock", confirm=True)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert Journal(rt.inst.journal_path).verify() >= 0
    rt.close()
