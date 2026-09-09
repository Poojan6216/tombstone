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
