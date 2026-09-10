"""7.7 — backup and replica: a filesystem snapshot of the Chroma persist dir and a pg_dump before
erasure; after erasure the subject is fully recoverable from either. Declared OUT_OF_SCOPE on
every receipt; this shows the declaration is not decorative."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import WORK, pg_dsn
from adversarial._harness import Pipeline
from corpus.build import load_corpus
from tombstone.util import fingerprint_bytes


def run(n_subjects: int) -> dict[str, Any]:
    docs = load_corpus()
    subjects = [f"S-{i:04d}" for i in range(1, min(n_subjects, 10) + 1)]
    p = Pipeline(WORK / "attacks" / "s77", backend="chroma")
    p.ingest([d for d in docs if d.subject in subjects], capture=True)
    snapshot = WORK / "attacks" / "s77-snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    shutil.copytree(p.store.path, snapshot)  # type: ignore[attr-defined]
    recovered_fs = 0
    oos_listed = 0
    fingerprints = []
    for s in subjects:
        from tombstone.commands.trace import run_trace
        from tombstone.model.artifacts import ArtifactKind

        t, _ = run_trace(p.rt, s, with_store_gaps=False)
        fps = [
            a.embedding_fingerprint
            for a in t.artifacts
            if a.kind is ArtifactKind.EMBED and a.embedding_fingerprint
        ]
        _code, data, _ = p.erase(s)
        oos_listed += int("database backups and snapshots" in data["out_of_scope"])
        fingerprints.append(fps)
    # the live store has nothing; the snapshot has everything
    live_dir = p.store.path  # type: ignore[attr-defined]
    for fps in fingerprints:
        snap_hit = any(_scan(snapshot, fingerprint_bytes(fp)) for fp in fps[:3])
        live_hit = any(_scan(live_dir, fingerprint_bytes(fp)) for fp in fps[:3])
        recovered_fs += int(snap_hit and not live_hit)
    p.close()
    pg = {"attempted": False}
    dsn = pg_dsn()
    if dsn and shutil.which("pg_dump"):
        pg = _pg_dump_case(dsn, docs, subjects[:3])
    return {
        "id": "7.7",
        "name": "backup and replica",
        "survives": "every vector, in the snapshot taken before erasure",
        "rate": recovered_fs / max(1, len(subjects)),
        "rate_text": f"filesystem snapshot of the Chroma dir: {recovered_fs}/{len(subjects)} subjects fully recoverable after a VERIFIED erasure"
        + (
            f"; pg_dump: {pg.get('recovered')}/{pg.get('subjects')} recoverable"
            if pg.get("attempted")
            else "; pg_dump: not run"
        ),
        "mitigation": f"none possible from inside the application: receipts listed 'database backups and snapshots' as OUT_OF_SCOPE in {oos_listed}/{len(subjects)} cases; backup retention policy is the operator's",
        "detail": {
            "filesystem": {"subjects": len(subjects), "recovered": recovered_fs},
            "pg_dump": pg,
        },
    }


def _scan(root: Path, pattern: bytes) -> bool:
    return any(f.is_file() and pattern in f.read_bytes() for f in root.rglob("*"))


def _pg_dump_case(dsn: str, docs: Any, subjects: list[str]) -> dict[str, Any]:
    """pg_dump before the erasure, pg_restore into a scratch database afterwards, and ask the
    restored table for the subject's rows by id: a backup restore brings them straight back."""
    import psycopg

    from _common import fresh_pg_database

    db = fresh_pg_database(dsn, "tomb_attack_s77")
    p = Pipeline(WORK / "attacks" / "s77-pg", backend="pgvector", pg_dsn=db)
    p.ingest([d for d in docs if d.subject in subjects], capture=True)
    dump = WORK / "attacks" / "s77.dump"
    subprocess.run(["pg_dump", "-Fc", "-f", str(dump), db], check=True, capture_output=True)
    keys_by_subject: dict[str, list[str]] = {}
    for s in subjects:
        from tombstone.commands.trace import run_trace
        from tombstone.model.artifacts import ArtifactKind

        t, _ = run_trace(p.rt, s, with_store_gaps=False)
        keys_by_subject[s] = [a.store_key for a in t.artifacts if a.kind is ArtifactKind.EMBED]
        p.erase(s)
    # the live table no longer has the rows; the restored backup does
    live_dsn = db
    restore_db = fresh_pg_database(dsn, "tomb_attack_s77_restore")
    subprocess.run(
        ["pg_restore", "--no-owner", "-d", restore_db, str(dump)], check=True, capture_output=True
    )
    recovered = 0
    live_leftover = 0
    with (
        psycopg.connect(restore_db, autocommit=True) as rc,
        psycopg.connect(live_dsn, autocommit=True) as lc,
    ):
        for _s, keys in keys_by_subject.items():
            n_restored = rc.execute(
                "SELECT count(*) FROM documents WHERE id = ANY(%s)", (keys,)
            ).fetchone()[0]
            n_live = lc.execute(
                "SELECT count(*) FROM documents WHERE id = ANY(%s)", (keys,)
            ).fetchone()[0]
            recovered += int(n_restored == len(keys) and len(keys) > 0)
            live_leftover += int(n_live > 0)
    p.close()
    return {
        "attempted": True,
        "subjects": len(subjects),
        "recovered": recovered,
        "live_leftover": live_leftover,
        "method": "pg_dump -Fc before erasure; pg_restore into a scratch database; SELECT by id",
    }
