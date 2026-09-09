"""``tombstone receipt`` and ``tombstone replay``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from tombstone.cli import EXIT_OK, EXIT_UNVERIFIED, command, emit
from tombstone.errors import ReplayMismatch, TombstoneError
from tombstone.model.status import Outcome, Receipt
from tombstone.receipt.ledger import Ledger
from tombstone.registry import Runtime


def render_receipt_summary(r: Receipt) -> str:
    lines = [
        f"receipt {r.receipt_id}   trace {r.trace_id}   reason {r.reason}",
        f"subject {r.subject.short}   scope {r.scope.tenant}   semantics v{r.semantics_version}",
        "",
        f"  {'store':<24} {'kind':<8} {'outcome':<12} {'level':<9} reason",
    ]
    for s in r.statuses:
        lines.append(
            f"  {s.artifact.store:<24} {s.artifact.kind.value:<8} {s.outcome.value:<12} {(s.level.value if s.level else '-'):<9} {s.reason[:70]}"
        )
    for s in r.needs_human:
        lines.append(
            f"  {s.artifact.store:<24} {s.artifact.kind.value:<8} {s.outcome.value:<12} {'-':<9} {s.reason[:70]}"
        )
    lines.append("")
    c = r.counts
    lines.append(
        f"  VERIFIED {c.get(Outcome.VERIFIED, 0)}   UNVERIFIED {c.get(Outcome.UNVERIFIED, 0)}   RESIDUAL {c.get(Outcome.RESIDUAL, 0)}"
        f"   OUT_OF_SCOPE {len(r.out_of_scope)}   NEEDS_HUMAN {c.get(Outcome.NEEDS_HUMAN, 0)}"
    )
    lines.append("  out of scope: " + "; ".join(r.out_of_scope))
    for n in r.notes:
        lines.append(f"  note: {n}")
    lines.append(
        "  A receipt is a record of what was done and checked. It is not a legal instrument."
    )
    return "\n".join(lines)


@command("receipt", "show a receipt")
def _setup_receipt(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
    p.add_argument("receipt_id", nargs="?", default=None, help="receipt id (default: latest)")
    p.add_argument("--path", default=None, help="a receipt JSON file instead of the ledger")
    p.add_argument("--list", action="store_true", help="list receipts in the ledger")

    def run(ns: argparse.Namespace) -> int:
        if ns.path:
            r = Receipt.from_dict(json.loads(Path(ns.path).read_text(encoding="utf-8")))
            emit(ns, render_receipt_summary(r), r.to_dict())
            return EXIT_OK
        rt = Runtime.load(ns.config)
        try:
            ledger = Ledger(rt.inst.ledger_path)
            receipts = ledger.receipts()
            if ns.list:
                lines = [
                    f"{r.receipt_id}  {r.trace_id}  {r.reason}  V{r.counts.get(Outcome.VERIFIED, 0)} U{r.counts.get(Outcome.UNVERIFIED, 0)} R{r.counts.get(Outcome.RESIDUAL, 0)}"
                    for r in receipts
                ]
                emit(
                    ns,
                    "\n".join(lines) or "no receipts",
                    {"receipts": [r.receipt_id for r in receipts]},
                )
                return EXIT_OK
            if not receipts:
                raise TombstoneError("no receipts in the ledger")
            found = receipts[-1] if ns.receipt_id is None else ledger.find(ns.receipt_id)
            if found is None:
                raise TombstoneError(f"no receipt {ns.receipt_id!r} in the ledger")
            r = found
        finally:
            rt.close()
        emit(ns, render_receipt_summary(r), r.to_dict())
        return EXIT_OK

    return run


@command("replay", "re-derive every receipt from the journal and assert equality")
def _setup_replay(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
    def run(ns: argparse.Namespace) -> int:
        from tombstone.receipt.replay import replay_ledger

        rt = Runtime.load(ns.config)
        try:
            report = replay_ledger(rt.inst.ledger_path, rt.inst.journal_path)
        finally:
            rt.close()
        lines = [f"replay: {report.matched}/{report.receipts} receipts re-derived identically"]
        lines.extend(f"  MISMATCH {m}" for m in report.mismatches)
        lines.extend(f"  VERSION  {m}" for m in report.version_mismatches)
        emit(
            ns,
            "\n".join(lines),
            {
                "receipts": report.receipts,
                "matched": report.matched,
                "mismatches": report.mismatches,
                "version_mismatches": report.version_mismatches,
            },
        )
        if not report.ok:
            raise ReplayMismatch((report.mismatches or report.version_mismatches)[0])
        return EXIT_OK if report.ok else EXIT_UNVERIFIED

    return run
