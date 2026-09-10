from __future__ import annotations

import json
from pathlib import Path

import pytest

from tombstone.chain import GENESIS, HashChain
from tombstone.erase.journal import Journal
from tombstone.errors import ChainBroken, LockTimeout
from tombstone.receipt.ledger import Ledger


def test_append_and_verify(tmp_path: Path) -> None:
    c = HashChain(tmp_path / "chain.jsonl")
    assert c.head() == GENESIS
    r0 = c.append("a", {"x": 1})
    r1 = c.append("b", {"y": [1, 2]})
    assert r0.prev == GENESIS and r1.prev == r0.hash and r1.seq == 1
    assert c.verify() == 2
    assert c.head() == r1.hash


@pytest.mark.parametrize("victim", [0, 3, 7])
def test_tamper_one_byte_breaks_chain_and_names_record(tmp_path: Path, victim: int) -> None:
    path = tmp_path / "chain.jsonl"
    c = HashChain(path)
    for i in range(10):
        c.append("rec", {"i": i, "payload": "abcdefghij"})
    lines = path.read_text().splitlines()
    d = json.loads(lines[victim])
    d["body"]["payload"] = "abcdefghiX"  # flip one byte of the body
    lines[victim] = json.dumps(d, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ChainBroken) as ei:
        c.verify()
    assert f"record {victim}" in str(ei.value)


def test_tamper_by_deleting_a_record(tmp_path: Path) -> None:
    path = tmp_path / "chain.jsonl"
    c = HashChain(path)
    for i in range(5):
        c.append("rec", {"i": i})
    lines = path.read_text().splitlines()
    del lines[2]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ChainBroken, match="record 2"):
        c.verify()


def test_tamper_hash_field(tmp_path: Path) -> None:
    path = tmp_path / "chain.jsonl"
    c = HashChain(path)
    c.append("rec", {"i": 0})
    c.append("rec", {"i": 1})
    lines = path.read_text().splitlines()
    d = json.loads(lines[1])
    d["hash"] = "0" * 64
    lines[1] = json.dumps(d, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ChainBroken, match="record 1"):
        c.verify()


def test_lock_contention_fails_loudly(tmp_path: Path) -> None:
    import fcntl
    import os

    path = tmp_path / "chain.jsonl"
    c = HashChain(path, lock_timeout_s=0.2)
    lock_path = path.with_suffix(".jsonl.lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        with pytest.raises(LockTimeout):
            c.append("x", {})
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    c.append("x", {})
    assert c.verify() == 1


def test_journal_resume_helpers(tmp_path: Path) -> None:
    j = Journal(tmp_path / "journal.jsonl")
    j.saga_start("S1", "T1", "h" * 64, "default", "dsr-1", 3)
    j.step_begin("S1", "suppress:chroma", "suppress", "chroma", ["a", "b"])
    j.step_end("S1", "suppress:chroma", ok=True)
    j.step_begin("S1", "reclaim:chroma", "reclaim", "chroma", ["a", "b"])
    assert set(j.completed_steps("S1")) == {"suppress:chroma"}
    assert set(j.begun_steps("S1")) == {"suppress:chroma", "reclaim:chroma"}
    assert j.open_sagas() == ["S1"]
    assert j.saga_for_trace("T1") == "S1"
    j.saga_end("S1", "R1")
    assert j.open_sagas() == []
    assert j.verify() == 5


def test_ledger_prev_receipt_hash(tmp_path: Path) -> None:
    led = Ledger(tmp_path / "ledger.jsonl")
    assert led.prev_receipt_hash() == GENESIS
    assert led.receipts() == []
    assert led.verify() == 0


# --- 5.6: what a signature on a receipt is actually worth ------------------------------------------


def _signed(key: object, **over: object) -> object:
    """A receipt signed by ``key``, with fields overridden."""
    from tombstone.model.artifacts import Scope, SubjectRef
    from tombstone.model.status import Receipt
    from tombstone.receipt.sign import public_key_hex, sign_bytes
    from tombstone.util import canonical_json

    base: dict[str, object] = {
        "receipt_id": "01R",
        "trace_id": "01T",
        "subject": SubjectRef("a" * 64),
        "scope": Scope("default"),
        "reason": "dsr-1",
        "statuses": [],
        "needs_human": [],
        "out_of_scope": ["backups"],
        "counts": {},
        "journal_head": "0" * 64,
        "prev_receipt_hash": "0" * 64,
        "semantics_version": "1",
        "created_ms": 1,
        "notes": [],
        "signature": "",
        "public_key": "",
    }
    base.update(over)
    r = Receipt(**base)  # type: ignore[arg-type]
    sig = sign_bytes(key, canonical_json(r.unsigned_payload()).encode("utf-8"))  # type: ignore[arg-type]
    return r.with_signature(sig, public_key_hex(key))  # type: ignore[arg-type]


def test_a_receipt_checked_against_its_own_key_is_not_evidence_of_origin() -> None:
    """A receipt carries the key that signed it, and the signature cannot cover that key.

    So anyone can write a receipt, sign it with a key they just generated, and it verifies. The
    verifier must not call that "OK" — it is integrity, not authenticity — and it must say so,
    because a receipt is the artifact an operator would hand to an auditor.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from tombstone.verify.independent import _check_signature

    operator = Ed25519PrivateKey.generate()
    attacker = Ed25519PrivateKey.generate()

    genuine = _signed(operator)
    forged = _signed(attacker, reason="forged-by-attacker", notes=["erasure complete"])

    # both verify against their own embedded key — that is the whole point of the finding
    for r in (genuine, forged):
        ok, trust = _check_signature(r, None)  # type: ignore[arg-type]
        assert ok, "a self-consistent receipt does verify against its own key"
        assert "not who signed it" in trust, trust
        assert "--public-key" in trust, "must say how to get a real answer"


def test_an_operator_supplied_key_rejects_the_forgery(tmp_path: Path) -> None:
    """With the operator's real public key, the forged receipt is caught."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from tombstone.verify.independent import _check_signature

    operator = Ed25519PrivateKey.generate()
    attacker = Ed25519PrivateKey.generate()
    pem = tmp_path / "public.pem"
    pem.write_bytes(
        operator.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )

    ok, trust = _check_signature(_signed(operator), pem)  # type: ignore[arg-type]
    assert ok and str(pem) in trust

    ok, trust = _check_signature(_signed(attacker, reason="forged"), pem)  # type: ignore[arg-type]
    assert not ok, "a receipt signed by another key must not verify against the operator's"
    assert "different key" in trust, trust
