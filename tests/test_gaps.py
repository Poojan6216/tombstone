"""1.6: data written around the wrapper is reported as unlineaged; stores with no capture named."""

from __future__ import annotations

from pathlib import Path

from tests import _stores
from tests.conftest import requires_chroma
from tombstone.lineage.capture import EmbedRecord
from tombstone.lineage.gaps import detect_gaps
from tombstone.model.artifacts import Scope


@requires_chroma
def test_direct_ingest_is_reported_as_gap(tmp_path: Path, pepper: bytes) -> None:
    store = _stores.make_backend("chroma", tmp_path)
    emb = _stores.embedder()
    # 20 docs written directly to Chroma, bypassing capture
    texts = [f"pre-existing document {i} about something" for i in range(20)]
    store._coll.add(
        ids=[f"old{i}" for i in range(20)],
        embeddings=emb.embed(texts),
        documents=texts,
        metadatas=[{"legacy": True} for _ in texts],
    )
    with _stores.lineage_and_capture(tmp_path) as (lineage, capture):
        capture.register(store.name, store.kind)
        report = detect_gaps(lineage, Scope("default"), {store.name: store})
        assert report.has_gaps
        cov = report.coverage[0]
        assert cov.total == 20 and cov.unlineaged_estimate == 20 and cov.coverage == 0.0
        assert store.name in report.stores_without_capture
        assert any("none with lineage" in m for m in report.messages)
        assert report.store_gaps() == ((store.name, 20),)
        # now capture 10 more through the wrapper: coverage rises, the 20 remain
        docs = _stores.stamped_docs(pepper, n_subjects=2, per_subject=1)[:10]
        t = [x for x, _ in docs]
        recs = capture.prepare_embeds(
            store.name,
            emb.name,
            [f"new{i}" for i in range(10)],
            emb.embed(t),
            [m for _, m in docs],
            t,
        )
        store.add(recs)
        report = detect_gaps(lineage, Scope("default"), {store.name: store})
        cov = report.coverage[0]
        assert cov.total == 30 and cov.unlineaged_estimate == 20
        assert store.name not in report.stores_without_capture
        assert report.store_gaps() == ((store.name, 20),)
    store.close()


def test_no_gaps_when_everything_captured(tmp_path: Path, pepper: bytes) -> None:
    store = _stores.make_backend("faiss", tmp_path)
    emb = _stores.embedder()
    docs = _stores.stamped_docs(pepper, n_subjects=2, per_subject=1)[:10]
    with _stores.lineage_and_capture(tmp_path) as (lineage, capture):
        capture.register(store.name, store.kind)
        t = [x for x, _ in docs]
        store.add(
            capture.prepare_embeds(
                store.name,
                emb.name,
                [f"k{i}" for i in range(10)],
                emb.embed(t),
                [m for _, m in docs],
                t,
            )
        )
        report = detect_gaps(lineage, Scope("default"), {store.name: store})
        assert not report.has_gaps and report.coverage[0].coverage == 1.0
    store.close()


def test_sampling_extrapolates(tmp_path: Path, pepper: bytes) -> None:
    store = _stores.make_backend("faiss", tmp_path)
    emb = _stores.embedder()
    with _stores.lineage_and_capture(tmp_path) as (lineage, capture):
        capture.register(store.name, store.kind)
        docs = _stores.stamped_docs(pepper, n_subjects=4, per_subject=1)[:20]
        t = [x for x, _ in docs]
        store.add(
            capture.prepare_embeds(
                store.name,
                emb.name,
                [f"a{i}" for i in range(20)],
                emb.embed(t),
                [m for _, m in docs],
                t,
            )
        )
        # 20 legacy rows written around capture
        legacy = [
            EmbedRecord(
                f"z{i}", emb.embed([f"legacy {i}"])[0], {"legacy": True}, f"legacy {i}", None, None
            )
            for i in range(20)
        ]  # type: ignore[arg-type]
        store._add(legacy)
        report = detect_gaps(lineage, Scope("default"), {store.name: store}, sample=10)
        cov = report.coverage[0]
        assert cov.sampled == 10 and cov.total == 40
        assert 0 < cov.unlineaged_estimate <= 40
    store.close()
