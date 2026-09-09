"""``tombstone trace`` and ``tombstone status``. Rendering only; the numbers come from the trace."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

from tombstone.cli import EXIT_OK, EXIT_UNVERIFIED, command, emit
from tombstone.lineage.gaps import GapReport, detect_gaps
from tombstone.lineage.trace import trace as pure_trace
from tombstone.model.artifacts import SubjectRef
from tombstone.model.lineage import Trace
from tombstone.registry import Runtime

RED = "\x1b[31m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"


def _colour(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


def run_trace(
    rt: Runtime, subject_raw: str, with_store_gaps: bool = True
) -> tuple[Trace, GapReport | None]:
    subject = SubjectRef.from_raw(subject_raw, rt.pepper())
    report: GapReport | None = None
    store_gaps: tuple[tuple[str, int], ...] = ()
    if with_store_gaps:
        try:
            stores = rt.all_stores()
        except Exception as e:
            stores = dict(rt._stores)
            store_gaps = (("<unopenable store>", -1),)
            sys.stderr.write(f"warning: could not open every store: {e}\n")
        report = detect_gaps(rt.lineage, rt.scope, stores)
        store_gaps = store_gaps + report.store_gaps()
    snap = rt.lineage.snapshot(rt.scope, store_gaps)
    t = pure_trace(subject, rt.scope, snap)
    rt.lineage.save_trace(t)
    return t, report


def render_trace(t: Trace) -> str:
    lines = [
        f"trace:    {t.trace_id}",
        f"subject:  {t.subject.short}   (raw id never logged)",
        f"scope:    {t.scope.tenant}",
        f"snapshot: sha256:{t.snapshot_hash[:16]}…",
        f"artifacts descending from this subject: {len(t.artifacts)}",
        "",
    ]
    by_store = t.by_store()
    shared = set(t.shared)
    tomb = set(t.already_tombstoned)
    for store in sorted(by_store):
        refs = by_store[store]
        kinds: dict[str, int] = {}
        for r in refs:
            kinds[r.kind.value] = kinds.get(r.kind.value, 0) + 1
        kind_str = ", ".join(f"{k}×{v}" for k, v in sorted(kinds.items()))
        flags = []
        if any(r.artifact_id in shared for r in refs):
            flags.append(f"shared:{sum(r.artifact_id in shared for r in refs)}")
        if any(r.artifact_id in tomb for r in refs):
            flags.append(f"already-tombstoned:{sum(r.artifact_id in tomb for r in refs)}")
        lines.append(
            f"  {store:<28} {len(refs):>4}   {kind_str}{'   ' + ' '.join(flags) if flags else ''}"
        )
    if t.third_party_hits:
        lines.append("")
        lines.append(f"third-party mentions (NEEDS_HUMAN, never erased): {len(t.third_party_hits)}")
        for a in t.third_party_hits:
            lines.append(f"  {a.kind.value} {a.artifact_id} in {a.store}")
    lines.append("")
    if t.gaps:
        lines.append(_colour(f"gaps: {len(t.gaps)} — erase will refuse without --accept-gaps", RED))
        for g in t.gaps:
            lines.append(_colour(f"  ! {g}", RED))
    else:
        lines.append("gaps: none")
    lines.append("")
    lines.append(f"next: tombstone erase --trace {t.trace_id} --reason <dsr-id> --confirm")
    return "\n".join(lines)


@command("trace", "list every artifact descending from a subject")
def _setup_trace(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
    p.add_argument("--subject", required=True, help="raw subject id (hashed immediately)")
    p.add_argument(
        "--no-store-scan", action="store_true", help="skip sampling stores for unlineaged entries"
    )

    def run(ns: argparse.Namespace) -> int:
        rt = Runtime.load(ns.config)
        try:
            t, _report = run_trace(rt, ns.subject, with_store_gaps=not ns.no_store_scan)
        finally:
            rt.close()
        emit(ns, render_trace(t), t.to_dict())
        return EXIT_UNVERIFIED if t.gaps else EXIT_OK

    return run


def status_payload(rt: Runtime) -> dict[str, Any]:
    from tombstone.erase.journal import Journal
    from tombstone.receipt.ledger import Ledger

    stores = rt.all_stores()
    report = detect_gaps(rt.lineage, rt.scope, stores)
    counts_by_store = rt.lineage.counts_by_store(rt.scope)
    pins = {name: pin.to_dict() for name, pin in rt.lineage.all_pins().items()}
    journal = Journal(rt.inst.journal_path)
    ledger = Ledger(rt.inst.ledger_path)
    dlq_depth = sum(1 for r in journal.records() if r.type == Journal.DLQ)
    return {
        "scope": rt.scope.tenant,
        "lineage": {
            "backend": rt.lineage.backend,
            "nodes_by_kind": rt.lineage.counts_by_kind(rt.scope),
            "subjects": rt.lineage.subject_count(rt.scope),
        },
        "stores": [
            {
                "name": c.store,
                "kind": stores[c.store].kind if c.store in stores else "?",
                "entries": c.total,
                "lineage_nodes": counts_by_store.get(c.store, 0),
                "coverage": round(c.coverage, 4),
                "unlineaged_estimate": c.unlineaged_estimate,
                "capabilities": sorted(x.value for x in stores[c.store].capabilities)
                if c.store in stores
                else [],
            }
            for c in report.coverage
        ],
        "gaps": list(report.messages),
        "pins": pins,
        "journal": {"records": len(journal.records()), "open_sagas": journal.open_sagas()},
        "dlq_depth": dlq_depth,
        "receipts": len(ledger.receipts()),
    }


def render_status(s: dict[str, Any]) -> str:
    lines = [
        f"scope: {s['scope']}   lineage: {s['lineage']['backend']}   subjects: {s['lineage']['subjects']}"
    ]
    kinds = s["lineage"]["nodes_by_kind"]
    lines.append("nodes: " + (", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "none"))
    lines.append("")
    lines.append(
        f"  {'store':<28} {'kind':<14} {'entries':>8} {'lineage':>8} {'coverage':>9}  capabilities"
    )
    for st in s["stores"]:
        cov = f"{st['coverage'] * 100:5.1f}%"
        cov_s = _colour(cov, RED) if st["coverage"] < 1.0 else cov
        lines.append(
            f"  {st['name']:<28} {st['kind']:<14} {st['entries']:>8} {st['lineage_nodes']:>8} {cov_s:>9}  {','.join(st['capabilities'])}"
        )
    if s["gaps"]:
        lines.append("")
        for g in s["gaps"]:
            lines.append(_colour(f"  ! {g}", RED))
    lines.append("")
    pins = s["pins"]
    lines.append(
        f"pins: {len(pins)}"
        + (
            ""
            if not pins
            else "  " + ", ".join(f"{k}({v['kind']})" for k, v in sorted(pins.items()))
        )
    )
    lines.append(
        f"journal: {s['journal']['records']} records, open sagas: {len(s['journal']['open_sagas'])}   dlq depth: {s['dlq_depth']}   receipts: {s['receipts']}"
    )
    return "\n".join(lines)


@command("status", "lineage coverage per store, pins, DLQ depth")
def _setup_status(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
    def run(ns: argparse.Namespace) -> int:
        rt = Runtime.load(ns.config)
        try:
            payload = status_payload(rt)
        finally:
            rt.close()
        emit(ns, render_status(payload), payload)
        return EXIT_UNVERIFIED if payload["gaps"] else EXIT_OK

    return run
