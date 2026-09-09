"""Phase 4 plumbing with a fake ModelOps: suppression excludes the shard, reclaim retrains only
that shard, probes read the state. No torch needed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests import _stores
from tombstone.model.artifacts import ArtifactKind, Scope, SubjectRef
from tombstone.model.status import VerifyLevel
from tombstone.stores.adapter import AdapterStore
from tombstone.stores.base import ProbeSet
from tombstone.train.dataset import DatasetStore, build_dataset


class FakeOps:
    """Writes fake adapter files whose bytes encode which examples they were trained on, so
    'weights changed' and 'extraction' can be simulated deterministically."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.memorised: dict[str, set[str]] = {}  # adapter dir name → texts it "knows"

    def _write(self, d: Path, texts: list[str]) -> None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
        (d / "adapter_model.safetensors").write_bytes(("|".join(sorted(texts))).encode())
        self.memorised[str(d)] = set(texts)

    def train_shard(
        self, dataset: DatasetStore, adapters_dir: Path, shard: int, cfg: Any
    ) -> dict[str, Any]:
        texts = [e.text for e in dataset.shard_examples(shard)]
        self._write(adapters_dir / f"shard-{shard:02d}", texts)
        self.calls.append(("train_shard", shard))
        return {"examples": len(texts)}

    def compose(
        self, adapters_dir: Path, serving_dir: Path, base_model: str, exclude: Any
    ) -> dict[str, Any]:
        texts: set[str] = set()
        for d in sorted(adapters_dir.glob("shard-*")):
            if int(d.name.split("-")[1]) in set(exclude):
                continue
            texts |= self.memorised.get(str(d), set())
        self._write(serving_dir, sorted(texts))
        self.calls.append(("compose", tuple(sorted(exclude))))
        return {"excluded": sorted(exclude)}

    def exact_unlearn(
        self, dataset: DatasetStore, adapters_dir: Path, serving_dir: Path, shard: int, cfg: Any
    ) -> dict[str, Any]:
        self.train_shard(dataset, adapters_dir, shard, cfg)
        self.compose(adapters_dir, serving_dir, "", [])
        self.calls.append(("exact_unlearn", shard))
        return {"wall_clock_s": 1.0}

    def approximate_unlearn(
        self, base_model: str, adapter_dir: Path, forget: Any, retain: Any, cfg: Any
    ) -> dict[str, Any]:
        known = self.memorised.get(str(adapter_dir), set())
        # approximate: forgets half of the forget set (the residual case, by construction)
        keep = set(list(sorted(forget))[: len(forget) // 2])
        self._write(adapter_dir, sorted((known - set(forget)) | keep))
        self.calls.append(("approx", cfg.method))
        return {"wall_clock_s": 2.0}

    def extraction(self, base_model: str, adapter_dir: Path, prompts: Any) -> tuple[int, int]:
        known = self.memorised.get(str(adapter_dir), set())
        hits = sum(
            1
            for prefix, expected in prompts
            if any(t.startswith(prefix) and expected in t for t in known)
        )
        return hits, len(prompts)

    def mia(
        self, base_model: str, adapter_dir: Path, members: Any, reference: Any
    ) -> dict[str, Any]:
        known = self.memorised.get(str(adapter_dir), set())
        frac = sum(1 for m in members if m in known) / max(1, len(members))
        auc = 0.5 + 0.45 * frac
        return {
            "loss": {"auc": auc, "ci_low": auc - 0.04, "ci_high": auc + 0.04},
            "mink": {"auc": auc, "ci_low": auc - 0.04, "ci_high": auc + 0.04},
        }

    def perplexity(self, base_model: str, adapter_dir: Path, texts: Any) -> float:
        return 10.0


def _setup(tmp_path: Path, pepper: bytes, shards: int = 4):
    docs = _stores.stamped_docs(pepper, n_subjects=6, per_subject=1)
    cm = _stores.lineage_and_capture(tmp_path)
    lineage, capture = cm.__enter__()
    chunks = [(capture.ensure_chunk(md, text)[0], text) for text, md in docs]
    manifest_path = tmp_path / "train" / "manifest.json"
    build_dataset(capture, "ft-dataset", manifest_path, chunks, shards=shards)
    ds = DatasetStore("ft-dataset", manifest_path)
    ops = FakeOps()
    for shard in range(shards):
        ops.train_shard(ds, tmp_path / "adapters", shard, None)
    ops.compose(tmp_path / "adapters", tmp_path / "adapters" / "serving", "", [])
    store = AdapterStore(
        "lora/support", tmp_path / "adapters", shards=shards, base_model="fake", dataset=ds, ops=ops
    )
    nodes = store.register_lineage(capture, ds)
    return lineage, capture, ds, ops, store, nodes, docs, cm


def test_register_creates_adapter_nodes_with_train_edges(tmp_path: Path, pepper: bytes) -> None:
    lineage, capture, ds, ops, store, nodes, docs, cm = _setup(tmp_path, pepper)
    try:
        assert store.capabilities == frozenset(
            {VerifyLevel.LOGICAL, VerifyLevel.MODEL, VerifyLevel.PHYSICAL}
        )
        assert {n.store_key for n in nodes} <= {f"shard-{i:02d}" for i in range(4)}
        for n in nodes:
            parents = lineage.edges_to([n.artifact_id])
            assert parents and all(e.via == f"adapter:{n.store_key}" for e in parents)
            assert all(lineage.node(e.parent).kind is ArtifactKind.TRAIN for e in parents)
        # idempotent
        again = store.register_lineage(capture, ds)
        assert {n.artifact_id for n in again} == {n.artifact_id for n in nodes}
    finally:
        cm.__exit__(None, None, None)


def test_suppress_excludes_shard_and_reclaim_retrains_only_it(
    tmp_path: Path, pepper: bytes
) -> None:
    from tombstone.lineage.trace import trace as pure_trace

    lineage, capture, ds, ops, store, nodes, docs, cm = _setup(tmp_path, pepper)
    try:
        subject = SubjectRef(docs[0][1]["tombstone.subject"])
        t = pure_trace(subject, Scope("default"), lineage.snapshot(Scope("default")))
        adapters = [a for a in t.artifacts if a.kind is ArtifactKind.ADAPTER]
        trains = [a for a in t.artifacts if a.kind is ArtifactKind.TRAIN]
        assert len(adapters) == 1 and trains
        shard = int(adapters[0].store_key.split("-")[1])
        store.set_context(t)
        # before: the subject's text is extractable from serving
        forget_texts = store._snapshot_forget_texts()
        assert forget_texts
        prompts = [(x[:20], x[20:]) for x in forget_texts]
        assert ops.extraction("", store.serving_dir, prompts)[0] == len(prompts)
        # phase 1: suppression recomposes serving without the shard → not extractable, reversible
        store.suppress(adapters)
        assert store.state()["excluded"] == [shard]
        assert ("compose", (shard,)) in ops.calls
        assert ops.extraction("", store.serving_dir, prompts)[0] == 0
        assert not store.probe_logical(
            adapters[0], ProbeSet(artifact_id=adapters[0].artifact_id)
        ).found
        # the shard adapter itself is untouched so far → physical says "unchanged"
        assert store.probe_physical(adapters[0]).found
        # dataset reclaim drops the rows (the saga does this before the adapter reclaims)
        ds.suppress(trains)
        ds.reclaim(trains)
        # phase 2: exact retrain of that shard only, then recomposed without exclusions
        r = store.reclaim(adapters)
        assert not r.noop and r.method.startswith(f"exact retrain: shard {shard}")
        assert [c for c in ops.calls if c[0] == "train_shard"][-1] == ("train_shard", shard)
        assert store.state()["excluded"] == []
        assert not store.probe_physical(adapters[0]).found
        pm = store.probe_model(adapters[0])
        assert pm["measurement"]["canary_rate"] == 0.0 and not pm["found"]
        # other subjects still extractable from serving (their shards untouched)
        other = [x for x, md in docs if md["tombstone.subject"] != subject.hmac][:3]
        assert ops.extraction("", store.serving_dir, [(x[:20], x[20:]) for x in other])[0] == 3
        store.finalize()
        assert store.state()["forget"] == {}
    finally:
        cm.__exit__(None, None, None)


def test_unsharded_adapter_is_residual_prone(tmp_path: Path, pepper: bytes) -> None:
    from tombstone.config import ModelConfig
    from tombstone.lineage.trace import trace as pure_trace

    docs = _stores.stamped_docs(pepper, n_subjects=4, per_subject=1)
    with _stores.lineage_and_capture(tmp_path) as (lineage, capture):
        chunks = [(capture.ensure_chunk(md, text)[0], text) for text, md in docs]
        manifest_path = tmp_path / "train" / "manifest.json"
        build_dataset(capture, "ft-dataset", manifest_path, chunks, shards=1)
        ds = DatasetStore("ft-dataset", manifest_path)
        ops = FakeOps()
        ops._write(tmp_path / "adapters" / "unsharded", [t for t, _ in docs])
        store = AdapterStore(
            "lora/flat",
            tmp_path / "adapters",
            shards=1,
            base_model="fake",
            dataset=ds,
            ops=ops,
            runtime_model_cfg=ModelConfig(unlearn="npo", unlearn_steps=5),
        )
        nodes = store.register_lineage(capture, ds)
        assert [n.store_key for n in nodes] == ["unsharded"]
        assert store.capabilities == frozenset({VerifyLevel.LOGICAL, VerifyLevel.PHYSICAL})
        subject = SubjectRef(docs[0][1]["tombstone.subject"])
        t = pure_trace(subject, Scope("default"), lineage.snapshot(Scope("default")))
        adapters = [a for a in t.artifacts if a.kind is ArtifactKind.ADAPTER]
        store.set_context(t)
        store.suppress(adapters)  # cannot hide an unsharded adapter; recorded only
        assert store.probe_logical(adapters[0], ProbeSet(artifact_id=adapters[0].artifact_id)).found
        r = store.reclaim(adapters)
        assert r.method.startswith("approximate unlearn (npo")
        pm = store.probe_model(adapters[0])
        assert 0 < pm["measurement"]["canary_rate"] < 1  # residual by construction
        assert pm["found"]


def test_exact_without_dataset_is_refused(tmp_path: Path) -> None:
    from tombstone.errors import NotSupported
    from tombstone.model.artifacts import ArtifactRef

    ops = FakeOps()
    ops._write(tmp_path / "adapters" / "shard-00", ["a"])
    ops._write(tmp_path / "adapters" / "shard-01", ["b"])
    store = AdapterStore("lora/x", tmp_path / "adapters", shards=2, base_model="fake", ops=ops)
    ref = ArtifactRef(
        "01A", ArtifactKind.ADAPTER, "lora/x", "shard-00", Scope("default"), "h" * 64, None
    )
    with pytest.raises(NotSupported, match="dataset"):
        store.reclaim([ref])
