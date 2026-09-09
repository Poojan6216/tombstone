"""1.5: sharded dataset with lineage; every subject in exactly one shard; manifest hash pins."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import _stores
from tombstone.errors import PinMismatch
from tombstone.model.artifacts import ArtifactKind, Scope
from tombstone.train.dataset import DatasetStore, build_dataset, manifest_hash, shard_for


def test_build_dataset_shards_by_subject(tmp_path: Path, pepper: bytes) -> None:
    docs = _stores.stamped_docs(pepper, n_subjects=6, per_subject=2)
    with _stores.lineage_and_capture(tmp_path) as (lineage, capture):
        chunks = []
        for text, md in docs:
            node, _ = capture.ensure_chunk(md, text)
            chunks.append((node, text))
        manifest = build_dataset(
            capture, "ft-dataset", tmp_path / "train" / "manifest.json", chunks, shards=4
        )
        assert manifest["manifest_hash"] == manifest_hash(manifest)
        assert len(manifest["examples"]) == len(docs)
        # every subject maps to exactly one shard
        by_subject: dict[str, set[int]] = {}
        for e in manifest["examples"]:
            by_subject.setdefault(e["subject"], set()).add(e["shard"])
            assert e["shard"] == shard_for(e["subject"], 4)
        assert all(len(s) == 1 for s in by_subject.values())
        # no content in the manifest
        raw = (tmp_path / "train" / "manifest.json").read_text()
        assert "says something unique" not in raw
        # TRAIN nodes + chunk→train edges tagged with the shard
        snap = lineage.snapshot(Scope("default"))
        trains = [n for n in snap.nodes if n.kind is ArtifactKind.TRAIN]
        assert len(trains) == len(docs)
        for t in trains:
            parents = lineage.edges_to([t.artifact_id])
            assert len(parents) == 1 and parents[0].via.startswith("train:shard-")
        # pin recorded
        pin = lineage.current_pin("ft-dataset")
        assert pin is not None and pin.to_dict()["manifest_hash"] == manifest["manifest_hash"]
        # round trip through the store view
        ds = DatasetStore("ft-dataset", tmp_path / "train" / "manifest.json")
        assert ds.count() == len(docs) and ds.shards() == 4
        total = sum(len(ds.shard_examples(s)) for s in range(4))
        assert total == len(docs)
        # rebuilding is idempotent (same TRAIN nodes, same hash)
        manifest2 = build_dataset(
            capture, "ft-dataset", tmp_path / "train" / "manifest.json", chunks, shards=4
        )
        assert manifest2["manifest_hash"] == manifest["manifest_hash"]


def test_modified_example_changes_manifest_hash_and_is_detected(
    tmp_path: Path, pepper: bytes
) -> None:
    docs = _stores.stamped_docs(pepper, n_subjects=2, per_subject=1)
    with _stores.lineage_and_capture(tmp_path) as (_lineage, capture):
        chunks = [(capture.ensure_chunk(md, text)[0], text) for text, md in docs]
        path = tmp_path / "train" / "manifest.json"
        m = build_dataset(capture, "ft-dataset", path, chunks, shards=2)
        blob = json.loads(path.read_text())
        blob["examples"][0]["content_hash"] = "0" * 64
        path.write_text(json.dumps(blob))
        with pytest.raises(PinMismatch, match="edited outside"):
            DatasetStore("ft-dataset", path).manifest()
        # a legitimate rewrite changes the hash
        blob["manifest_hash"] = manifest_hash(blob)
        path.write_text(json.dumps(blob))
        assert DatasetStore("ft-dataset", path).manifest()["manifest_hash"] != m["manifest_hash"]


def test_dataset_store_suppress_reclaim_probe(tmp_path: Path, pepper: bytes) -> None:
    docs = _stores.stamped_docs(pepper, n_subjects=3, per_subject=1)
    with _stores.lineage_and_capture(tmp_path) as (lineage, capture):
        chunks = [(capture.ensure_chunk(md, text)[0], text) for text, md in docs]
        path = tmp_path / "train" / "manifest.json"
        build_dataset(capture, "ft-dataset", path, chunks, shards=3)
        ds = DatasetStore("ft-dataset", path)
        target = next(
            n for n in lineage.snapshot(Scope("default")).nodes if n.kind is ArtifactKind.TRAIN
        )
        from tombstone.stores.base import ProbeSet

        ps = ProbeSet(artifact_id=target.artifact_id)
        assert ds.probe_logical(target.ref(), ps).found
        assert ds.probe_physical(target.ref()).found
        ds.suppress([target.ref()])
        assert not ds.probe_logical(target.ref(), ps).found  # hidden from the trainer
        assert ds.probe_physical(target.ref()).found  # but the bytes are still there
        r = ds.reclaim([target.ref()])
        assert not r.noop and r.measurement["deleted"] == 1
        assert not ds.probe_physical(target.ref()).found
        assert ds.reclaim([target.ref()]).noop
        assert ds.count() == len(docs) - 1
