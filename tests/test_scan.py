"""``tombstone scan`` — the command for someone who has a deletion request today and did not
install this tool three years ago.

The measurements are the easy part. What these tests mostly guard is the claim: the command tells
an operator it is safe to point at production, so "no insert, update or delete" has to be true,
and the separate fact that a database engine writes to its own files when you open it has to be
reported rather than glossed over. A scan that quietly rewrote somebody's index while printing
"read-only" would be exactly the kind of unearned assurance this project exists to refuse.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import _pipeline, _stores
from tombstone.scan import _census, _redact, render, scan


def _unmanaged_chroma(root: Path, n: int = 120, dupes: int = 10) -> Path:
    """A Chroma store built the ordinary way: no capture hook, no stamps, no lineage."""
    _stores.skip_unless("chroma")
    import chromadb
    from chromadb.config import Settings

    path = root / "chroma_db"
    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
    coll = client.get_or_create_collection(
        "customer-kb", metadata={"hnsw:space": "cosine"}, embedding_function=None
    )
    emb = _stores.embedder()
    texts = [f"support ticket {i}" for i in range(n)]
    vectors = emb.embed(texts)
    boiler = vectors[0]
    for i in range(1, dupes + 1):  # a boilerplate paragraph repeated across records
        vectors[i] = list(boiler)
    coll.upsert(
        ids=[f"doc-{i}" for i in range(n)],
        embeddings=vectors,
        documents=texts,
        metadatas=[{"source": f"crm/{i}.pdf"} for i in range(n)],
    )
    return path


def test_it_reports_a_store_it_has_never_seen(tmp_path: Path) -> None:
    _unmanaged_chroma(tmp_path)
    report = scan(root=tmp_path)

    assert len(report.stores) == 1, [s.name for s in report.stores]
    s = report.stores[0]
    assert s.kind == "chroma" and s.entries == 120
    assert s.sampled > 0
    assert s.stamped == 0 and s.coverage == 0.0, "nothing here arrived through Tombstone"
    assert s.physical, "a readable Chroma directory can be checked at the byte level"
    assert not report.any_traceable

    text = render(report)
    assert "nothing here can be traced to a person" in text
    assert "0%" in text


def test_it_finds_the_bytes_no_scan_could_ever_attribute(tmp_path: Path) -> None:
    """Duplicate content is the honest limit of byte-level proof, and it is worth knowing about
    before a request arrives rather than in the middle of one."""
    _unmanaged_chroma(tmp_path, n=120, dupes=10)
    report = scan(root=tmp_path, sample=120)
    s = report.stores[0]
    # 11 vectors share one fingerprint (the original plus ten copies)
    assert s.duplicate_sampled == 11, (s.duplicate_sampled, s.sampled)
    assert 0 < (s.duplicate_rate or 0) < 1
    assert "byte-identical to another entry" in render(report)


def test_the_no_write_claim_is_true(tmp_path: Path) -> None:
    """The scan says it issues no insert, update or delete. Check the data, not the wording:
    same count, same ids, same content afterwards."""
    import chromadb
    from chromadb.config import Settings

    path = _unmanaged_chroma(tmp_path)

    def contents() -> tuple[int, list[str], list[str]]:
        client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        coll = client.get_collection("customer-kb")
        got = coll.get(include=["documents"])
        client.clear_system_cache()
        return coll.count(), sorted(map(str, got["ids"])), sorted(got["documents"] or [])

    before = contents()
    scan(root=tmp_path)
    after = contents()
    assert before == after, "the scan changed the data it was only asked to look at"


def test_it_does_not_pretend_the_files_are_untouched(tmp_path: Path) -> None:
    """Chroma rewrites its index header and SQLite file on *any* open, by any client. The scan
    does not control that, so it must measure and report it rather than claim otherwise."""
    path = _unmanaged_chroma(tmp_path)
    report = scan(root=tmp_path)
    text = render(report)

    assert "no insert, update or delete" in text
    # whatever the engine did, the report and the filesystem must agree
    census_now = _census(tmp_path)
    again_before = dict(census_now)
    scan(root=tmp_path)
    again_after = _census(tmp_path)
    actually_changed = {p for p in again_after if again_before.get(p) != again_after[p]}

    if report.engine_wrote:
        assert "rewrote" in text
        assert all(Path(p).exists() for p in report.engine_wrote)
    else:
        assert "nothing on disk changed" in text
    # the claim must not be weaker than reality: if files move, the report has to say so
    if actually_changed:
        assert report.engine_wrote, (
            "files changed during a scan but the report claimed nothing did: "
            f"{sorted(actually_changed)[:3]}"
        )
    assert path.is_dir()


def test_a_managed_store_shows_its_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half: a store Tombstone captured should come back stamped, so the same command
    tells an existing user how complete their coverage is."""
    monkeypatch.chdir(tmp_path)
    _stores.skip_unless("chroma")
    h = _pipeline.build(tmp_path, ["chroma"], rag_subjects=("S-0001",))
    h["rt"].close()

    report = scan(root=tmp_path, config=h["cfg_path"])
    chroma = [s for s in report.stores if s.kind == "chroma"]
    assert chroma, [s.name for s in report.stores]
    assert chroma[0].stamped > 0 and (chroma[0].coverage or 0) > 0.9
    assert report.any_traceable
    assert "some of this is traceable" in render(report)


def test_a_password_never_reaches_the_report() -> None:
    """Scan output is the kind of thing people paste into a ticket."""
    out = _redact("postgresql://admin:hunter2@db.internal:5432/prod")
    assert "hunter2" not in out and "admin" in out and "db.internal" in out
    assert _redact("/var/lib/chroma") == "/var/lib/chroma"


def test_an_empty_directory_says_so_rather_than_failing(tmp_path: Path) -> None:
    report = scan(root=tmp_path)
    assert report.stores == []
    assert any("nothing found" in n for n in report.notes)
    assert "nothing found" in render(report)


def test_the_cli_exits_two_when_nothing_can_be_traced(tmp_path: Path) -> None:
    """Exit 2 is this tool's "checked, and the answer is not a clean pass" — the same code trace
    and erase use. An untraceable store is a finding, not a crash."""
    _unmanaged_chroma(tmp_path, n=30)
    p = subprocess.run(
        [sys.executable, "-m", "tombstone", "scan", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
        env={**os.environ, "TOMBSTONE_LOG_LEVEL": "error"},
    )
    assert p.returncode == 2, p.stderr
    payload: dict[str, Any] = json.loads(p.stdout)
    assert payload["any_traceable"] is False
    assert payload["stores"][0]["entries"] == 30
    assert payload["stores"][0]["coverage"] == 0.0
    # content never leaves the store
    assert "support ticket" not in p.stdout
