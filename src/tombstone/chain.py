"""Append-only, hash-chained JSONL with file locking. The ledger and the journal are both this.

Record layout (one JSON object per line)::

    {"seq": 12, "prev": "<sha256 of record 11>", "ts": 1725000000000,
     "type": "step_begin", "body": {...}, "hash": "<sha256 of canonical(seq, prev, ts, type, body)>"}

Record 0 has ``prev`` = 64 zeros. ``verify()`` recomputes every hash and every link and names
the first record index that fails (Hard Rule 6/9). Bodies never contain content (Hard Rule 7).
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tombstone.errors import ChainBroken, LockTimeout
from tombstone.util import canonical_json, sha256_hex, utc_ms

GENESIS = "0" * 64


@dataclass(frozen=True, slots=True)
class Record:
    seq: int
    prev: str
    ts: int
    type: str
    body: dict[str, Any]
    hash: str

    @staticmethod
    def compute_hash(seq: int, prev: str, ts: int, type_: str, body: dict[str, Any]) -> str:
        return sha256_hex(
            canonical_json({"seq": seq, "prev": prev, "ts": ts, "type": type_, "body": body})
        )

    def to_json(self) -> str:
        return canonical_json(
            {
                "seq": self.seq,
                "prev": self.prev,
                "ts": self.ts,
                "type": self.type,
                "body": self.body,
                "hash": self.hash,
            }
        )

    @staticmethod
    def from_json(line: str) -> Record:
        d = json.loads(line)
        return Record(
            seq=int(d["seq"]),
            prev=str(d["prev"]),
            ts=int(d["ts"]),
            type=str(d["type"]),
            body=dict(d["body"]),
            hash=str(d["hash"]),
        )


@contextmanager
def file_lock(path: Path, timeout_s: float) -> Iterator[None]:
    """Exclusive advisory lock on ``<path>.lock`` with a timeout; contention fails loudly."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise LockTimeout(
                        f"could not lock {path} within {timeout_s:.1f}s; another tombstone "
                        "process is writing to it"
                    ) from None
                time.sleep(0.01)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class HashChain:
    def __init__(self, path: Path, lock_timeout_s: float = 30.0) -> None:
        self.path = Path(path)
        self.lock_timeout_s = lock_timeout_s

    def read(self) -> list[Record]:
        if not self.path.is_file():
            return []
        out: list[Record] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(Record.from_json(line))
                except (ValueError, KeyError) as e:
                    raise ChainBroken(f"{self.path}: record {i} is not a valid record: {e}") from e
        return out

    def head(self) -> str:
        recs = self.read()
        return recs[-1].hash if recs else GENESIS

    def __len__(self) -> int:
        return len(self.read())

    def append(self, type_: str, body: dict[str, Any]) -> Record:
        """Append one record under the lock, fsync, return it."""
        with file_lock(self.path, self.lock_timeout_s):
            recs = self.read()
            seq = recs[-1].seq + 1 if recs else 0
            prev = recs[-1].hash if recs else GENESIS
            ts = utc_ms()
            rec = Record(
                seq, prev, ts, type_, body, Record.compute_hash(seq, prev, ts, type_, body)
            )
            line = rec.to_json() + "\n"
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            return rec

    def verify(self) -> int:
        """Return the record count. Raise ``ChainBroken`` naming the first bad record."""
        prev = GENESIS
        recs = self.read()
        for expected_seq, rec in enumerate(recs):
            if rec.seq != expected_seq:
                raise ChainBroken(
                    f"{self.path}: record {expected_seq} has seq {rec.seq} (expected {expected_seq})"
                )
            if rec.prev != prev:
                raise ChainBroken(
                    f"{self.path}: record {rec.seq} prev-hash does not link to record {rec.seq - 1}"
                )
            recomputed = Record.compute_hash(rec.seq, rec.prev, rec.ts, rec.type, rec.body)
            if recomputed != rec.hash:
                raise ChainBroken(f"{self.path}: record {rec.seq} hash mismatch (tampered)")
            prev = rec.hash
        return len(recs)
