"""The receipt ledger: a hash chain whose records are receipts (Hard Rule 7: never content)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tombstone.chain import GENESIS, HashChain, Record
from tombstone.model.status import Receipt


class Ledger:
    RECEIPT = "receipt"

    def __init__(self, path: Path, lock_timeout_s: float = 30.0) -> None:
        self.chain = HashChain(path, lock_timeout_s)
        self.path = Path(path)

    def prev_receipt_hash(self) -> str:
        recs = [r for r in self.chain.read() if r.type == self.RECEIPT]
        return recs[-1].hash if recs else GENESIS

    def append(self, receipt: Receipt) -> Record:
        return self.chain.append(self.RECEIPT, receipt.to_dict())

    def receipts(self) -> list[Receipt]:
        return [Receipt.from_dict(r.body) for r in self.chain.read() if r.type == self.RECEIPT]

    def records(self) -> list[Record]:
        return self.chain.read()

    def verify(self) -> int:
        return self.chain.verify()

    def find(self, receipt_id: str) -> Receipt | None:
        for r in self.chain.read():
            if r.type == self.RECEIPT and r.body.get("receipt_id") == receipt_id:
                return Receipt.from_dict(r.body)
        return None

    def head(self) -> str:
        return self.chain.head()

    def summary(self) -> dict[str, Any]:
        recs = self.chain.read()
        return {"records": len(recs), "head": recs[-1].hash if recs else GENESIS}
