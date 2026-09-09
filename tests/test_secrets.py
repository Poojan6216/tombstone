"""Hard Rule 7. Mandatory. Never xfail, never skip.

Seeds every secret format through everything that writes to disk under .tombstone/ and asserts
none of them appears in the ledger, the journal, the DLQ, any receipt, the lineage db, or the
captured logs. Later phases extend ``_exercise`` with stamp → trace → erase → receipt; the
assertion never changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tombstone.config import Installation
from tombstone.erase.journal import Journal
from tombstone.lineage.store import LineageStore
from tombstone.logging import get_logger, log
from tombstone.model.artifacts import ArtifactKind, Scope, SubjectRef
from tombstone.model.lineage import Edge, Node
from tombstone.receipt.ledger import Ledger
from tombstone.util import content_hash, new_ulid

FIXTURE = Path(__file__).parent / "fixtures" / "secrets" / "seeded.json"

pytestmark = pytest.mark.mandatory


def load_secrets() -> list[str]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    values = [s["value"] for s in data["secrets"]]
    assert len(values) >= 15
    return values


def _exercise_phase0(
    inst: Installation, secrets: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    pepper = inst.read_pepper()
    store = LineageStore.open_sqlite(inst.lineage_db_path)
    journal = Journal(inst.journal_path)
    ledger = Ledger(inst.ledger_path)
    logger = get_logger("test")
    for i, secret in enumerate(secrets):
        subject = SubjectRef.from_raw(secret, pepper)  # every secret doubles as a raw subject id
        src = Node(
            artifact_id=new_ulid(),
            kind=ArtifactKind.SOURCE,
            store="docs",
            store_key=f"doc-{i}",
            scope=Scope("default"),
            content_hash=content_hash(secret),  # content is only ever hashed
            embedding_fingerprint=None,
            subject_hmac=subject.hmac,
            created_seq=store.next_seq(),
        )
        chunk = Node(
            artifact_id=new_ulid(),
            kind=ArtifactKind.CHUNK,
            store="docs",
            store_key=f"doc-{i}#0",
            scope=Scope("default"),
            content_hash=content_hash("chunk: " + secret),
            embedding_fingerprint=None,
            subject_hmac=subject.hmac,
            created_seq=store.next_seq(),
        )
        store.add_nodes([src, chunk])
        store.add_edge(Edge(src.artifact_id, chunk.artifact_id, "chunk"))
        store.tombstone([chunk.artifact_id], "test", None)
        journal.saga_start("S" + str(i), "T" + str(i), subject.hmac, "default", "dsr-test", 2)
        journal.step_begin("S" + str(i), "suppress:docs", "suppress", "docs", [chunk.artifact_id])
        journal.step_end("S" + str(i), "suppress:docs", ok=True)
        ledger.chain.append("note", {"subject": subject.hmac, "artifact_ids": [chunk.artifact_id]})
        log(logger, logging.INFO, "stamped", subject=subject.short, artifact_id=src.artifact_id)
    store.close()


def _all_state_bytes(inst: Installation) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for p in inst.root.rglob("*"):
        if p.is_file():
            out[str(p.relative_to(inst.root))] = p.read_bytes()
    return out


def assert_no_secret_anywhere(inst: Installation, secrets: list[str], captured_logs: str) -> None:
    blobs = _all_state_bytes(inst)
    assert blobs, "nothing was written under .tombstone/ — the test is not exercising anything"
    for secret in secrets:
        needle = secret.encode("utf-8")
        for name, data in blobs.items():
            assert needle not in data, f"secret {secret!r} leaked into {name}"
            # also check common alternate encodings that a lazy serializer might use
            assert needle.hex().encode() not in data, f"hex of {secret!r} leaked into {name}"
        assert secret not in captured_logs, f"secret {secret!r} leaked into logs"


def test_no_secret_in_state_or_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    secrets = load_secrets()
    inst = Installation(tmp_path / ".tombstone")
    inst.ensure()
    logger = get_logger()
    logger.setLevel(logging.DEBUG)
    with caplog.at_level(logging.DEBUG, logger="tombstone"):
        _exercise_phase0(inst, secrets, caplog)
    assert_no_secret_anywhere(inst, secrets, caplog.text)


def test_fixture_has_fifteen_plus_formats() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    formats = {s["format"] for s in data["secrets"]}
    assert len(formats) >= 15
