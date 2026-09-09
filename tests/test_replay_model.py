"""The model leg through the whole pipeline with a fake trainer: saga → receipt → replay →
independent verifier. Covers the model-facts path in replay, which the real-model test is too
slow to exercise on every run."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import _stores
from tests.test_adapter_store import FakeOps
from tombstone.commands.erase import run_erase
from tombstone.commands.trace import run_trace
from tombstone.model.artifacts import ArtifactKind
from tombstone.model.status import Outcome
from tombstone.receipt.replay import assert_replay, replay_ledger
from tombstone.train.dataset import DatasetStore, build_dataset
from tombstone.verify.independent import verify_receipt_independently

CONFIG = """\
version: 1
scope: default
lineage: {{ backend: sqlite, path: {root}/.tombstone/lineage.db }}
stores:
  - {{ name: "docs", kind: docstore, path: {root}/docs.sqlite }}
  - {{ name: "ft-dataset", kind: dataset, manifest: {root}/train/manifest.json }}
  - {{ name: "lora/support", kind: adapter, path: {root}/adapters, shards: 4, base_model: fake, dataset: ft-dataset }}
model:
  base: fake
  unlearn: exact
out_of_scope:
  - "database backups and snapshots"
"""


def _build(tmp_path: Path, pepper: bytes):
    from tombstone.commands.init import run_init
    from tombstone.registry import Runtime
    from tombstone.stores.adapter import AdapterStore

    run_init(tmp_path)
    cfg = tmp_path / "tombstone.yaml"
    cfg.write_text(CONFIG.format(root=tmp_path))
    rt = Runtime.shared(cfg)
    capture = rt.capture()
    # stamp with the installation pepper, not the fixture's: the HMACs must match what a trace
    # computes from disk
    docs = _stores.stamped_docs(rt.pepper(), n_subjects=6, per_subject=1)
    chunks = []
    docstore = rt.store("docs")
    for text, md in docs:
        src = capture.ensure_source(md, text)
        docstore.put(src.artifact_id, "source", text, md)  # type: ignore[attr-defined]
        node, _ = capture.ensure_chunk(md, text)
        docstore.put(node.artifact_id, "chunk", text, md)  # type: ignore[attr-defined]
        chunks.append((node, text))
    build_dataset(capture, "ft-dataset", tmp_path / "train" / "manifest.json", chunks, shards=4)
    ds = DatasetStore("ft-dataset", tmp_path / "train" / "manifest.json")
    ops = FakeOps()
    for shard in range(4):
        ops.train_shard(ds, tmp_path / "adapters", shard, None)
    ops.compose(tmp_path / "adapters", tmp_path / "adapters" / "serving", "", [])
    store = rt.store("lora/support")
    assert isinstance(store, AdapterStore)
    store._ops = ops  # noqa: SLF001
    store.capabilities = store.detect_capabilities()
    store.register_lineage(capture, ds)
    return rt, cfg, ops, docs


def test_model_leg_replays_and_verifies_independently(
    tmp_path: Path, pepper: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    rt, cfg, ops, docs = _build(tmp_path, pepper)
    subject = "S-0000"
    t, _ = run_trace(rt, subject, with_store_gaps=False)
    kinds = {a.kind for a in t.artifacts}
    assert ArtifactKind.ADAPTER in kinds and ArtifactKind.TRAIN in kinds
    code, text, data = run_erase(rt, t.trace_id, "dsr-model", confirm=True)
    adapter_rows = [s for s in data["statuses"] if s["artifact"]["kind"] == "adapter"]
    assert adapter_rows, data["statuses"]
    row = adapter_rows[0]
    assert row["outcome"] == Outcome.VERIFIED.value and row["level"] == "model", row
    assert row["measurement"]["canary_total"] > 0, "a VERIFIED(model) row must carry evidence"
    assert code == 0, text
    # replay re-derives the model facts from the journal alone
    report = replay_ledger(rt.inst.ledger_path, rt.inst.journal_path)
    assert report.ok and report.matched == report.receipts == 1, report.mismatches
    assert_replay(rt.inst.ledger_path, rt.inst.journal_path)
    receipt_path = rt.inst.receipts_dir / f"{data['receipt_id']}.json"
    rt.close()
    res = verify_receipt_independently(
        receipt_path, rt.inst.public_key_path, rt.inst.ledger_path, cfg
    )
    assert res["ok"], res["text"]
    assert any(r["store"] == "lora/support" for r in res["rows"])


def test_adapter_without_evidence_is_unverified(
    tmp_path: Path, pepper: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hard Rule 2: no extraction prompts and no MIA reference must not read as VERIFIED."""
    monkeypatch.chdir(tmp_path)
    rt, cfg, ops, docs = _build(tmp_path, pepper)
    store = rt.store("lora/support")
    monkeypatch.setattr(store, "_snapshot_forget_texts", lambda: [])  # the app kept no examples
    t, _ = run_trace(rt, "S-0001", with_store_gaps=False)
    code, text, data = run_erase(rt, t.trace_id, "dsr-blind", confirm=True)
    adapter_rows = [s for s in data["statuses"] if s["artifact"]["kind"] == "adapter"]
    assert adapter_rows and all(s["outcome"] == Outcome.UNVERIFIED.value for s in adapter_rows), (
        adapter_rows
    )
    assert all(s["rule_id"] == "model_unprobed" for s in adapter_rows)
    assert "mia_reference" in adapter_rows[0]["reason"]
    assert code == 2
    assert replay_ledger(rt.inst.ledger_path, rt.inst.journal_path).ok
    rt.close()
