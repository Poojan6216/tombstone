"""Chroma's local HNSW segment persists its whole allocated buffer, uninitialised slots included
(see the ``tombstone.stores.chroma`` docstring). On Linux CI the physical probe found an erased
vector in those slots after a clean reclaim. These tests pin the two points where the adapter
zeroes them — after the rewrite, and at open after Chroma's init has written the buffer — and
the header decode both rely on."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import _stores
from tombstone.model.artifacts import ArtifactRef
from tombstone.stores.chroma import ChromaStore, _hnsw_header
from tombstone.util import fingerprint_bytes


def _segments(path: Path) -> list[Path]:
    return sorted(d for d in path.iterdir() if d.is_dir() and (d / "header.bin").is_file())


def _unused_region(seg: Path) -> bytes:
    h = _hnsw_header(seg)
    assert h is not None, seg
    data = (seg / "data_level0.bin").read_bytes()
    return data[h["offset_level0"] + h["count"] * h["per_element"] :]


def _build(tmp_path: Path, pepper: bytes, n: int = 60) -> tuple[ChromaStore, list[ArtifactRef]]:
    _stores.skip_unless("chroma")
    store = _stores.make_backend("chroma", tmp_path)
    assert isinstance(store, ChromaStore)
    emb = _stores.embedder()
    docs = _stores.stamped_docs(pepper, n_subjects=n, per_subject=1)[:n]
    with _stores.lineage_and_capture(tmp_path) as (_lineage, capture):
        texts = [x for x, _ in docs]
        recs = capture.prepare_embeds(
            store.name,
            emb.name,
            [f"k{i}" for i in range(n)],
            emb.embed(texts),
            [m for _, m in docs],
            texts,
        )
        store.add(recs)
        return store, [r.embed_node.ref() for r in recs]


def test_header_decode_matches_the_layout_chroma_wrote(tmp_path: Path, pepper: bytes) -> None:
    store, _ = _build(tmp_path, pepper)
    segs = _segments(store.path)
    assert segs, "Chroma wrote no segment directory"
    h = _hnsw_header(segs[0])
    assert h is not None
    # size_data_per_element = links (4 + 4*maxM0) + data (4*dims) + label (8)
    assert h["offset_data"] == 4 + 4 * h["max_m0"]
    assert h["label_offset"] == h["offset_data"] + 4 * _stores.DIMS
    assert h["per_element"] == h["label_offset"] + 8
    assert h["count"] <= h["max_elements"]
    assert (segs[0] / "data_level0.bin").stat().st_size == h["max_elements"] * h["per_element"]
    store.close()


def test_open_zeroes_what_chroma_init_wrote(tmp_path: Path, pepper: bytes) -> None:
    """Opening the store forces Chroma's init and then zeroes the slots past the live elements,
    so the file never carries this process's heap. Before the scrub, the region is heap garbage."""
    store, refs = _build(tmp_path, pepper)
    for seg in _segments(store.path):
        assert not any(_unused_region(seg)), "open left non-zero bytes in unused slots"
    store.close()
    # Plant what a Linux allocator can leave there — an erased vector's first 32 dims — beyond
    # the live count, then reopen: Chroma's init rewrites the file, the adapter zeroes it again.
    victim = refs[0]
    assert victim.embedding_fingerprint
    pattern = fingerprint_bytes(victim.embedding_fingerprint)
    for seg in _segments(tmp_path / "chroma"):
        h = _hnsw_header(seg)
        assert h is not None
        start = h["offset_level0"] + h["count"] * h["per_element"]
        with (seg / "data_level0.bin").open("r+b") as fh:
            fh.seek(start + 16)
            fh.write(pattern)
        assert pattern in (seg / "data_level0.bin").read_bytes()
    reopened = _stores.make_backend("chroma", tmp_path)
    assert isinstance(reopened, ChromaStore)
    for seg in _segments(reopened.path):
        assert not any(_unused_region(seg))
    assert all(pattern not in _unused_region(seg) for seg in _segments(reopened.path))
    assert reopened.count() == len(refs)
    reopened.close()


def test_reclaim_zeroes_unused_slots_and_reports_it(tmp_path: Path, pepper: bytes) -> None:
    store, refs = _build(tmp_path, pepper)
    victims = refs[:10]
    store.suppress(victims)
    res = store.reclaim(victims)
    assert not res.noop
    # the count is what the allocator happened to leave non-zero, so it is reported, not asserted
    assert "unused_slot_bytes_zeroed" in res.measurement, res
    assert res.measurement["segments_not_scrubbed"] == 0, res.detail
    for seg in _segments(store.path):
        assert not any(_unused_region(seg)), seg
    for v in victims:
        pr = store.probe_physical(v)
        assert not pr.found, (v.artifact_id, pr.locations, pr.measurement)
    # survivors are still served and still physically present
    assert store.count() == len(refs) - len(victims)
    assert store.probe_physical(refs[-1]).found
    store.close()


@pytest.mark.parametrize("bad", ["short", "version"])
def test_unrecognised_header_is_left_alone(tmp_path: Path, pepper: bytes, bad: str) -> None:
    """A header the decoder does not recognise means no scrub and a reported skip — never a guess."""
    store, refs = _build(tmp_path, pepper)
    seg = _segments(store.path)[0]
    hb = bytearray((seg / "header.bin").read_bytes())
    if bad == "short":
        (seg / "header.bin").write_bytes(bytes(hb[:-4]))
    else:
        hb[0] = 9  # PERSISTENCE_VERSION
        (seg / "header.bin").write_bytes(bytes(hb))
    zeroed, skipped = store._scrub_unused_slots()
    assert zeroed == 0 and len(skipped) == 1 and seg.name in skipped[0]
