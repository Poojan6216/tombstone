"""``tombstone erase`` and ``tombstone dlq``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from tombstone.cli import EXIT_ERROR, EXIT_OK, command, emit
from tombstone.erase.journal import Journal
from tombstone.erase.saga import Saga, SagaOptions
from tombstone.errors import ConfirmationRequired, SagaError
from tombstone.receipt.render import render_receipt
from tombstone.registry import Runtime


def run_erase(
    rt: Runtime,
    trace_id: str,
    reason: str,
    confirm: bool,
    accept_gaps: bool = False,
    reclaim: bool = True,
    semantic: bool = False,
    retry: bool = False,
) -> tuple[int, str, dict[str, object]]:
    if rt.cfg.erase.require_confirm and not confirm:
        raise ConfirmationRequired(
            "erase is destructive and requires explicit confirmation: pass --confirm "
            "(or confirm the MCP elicitation). Nothing was changed."
        )
    if not reason.strip():
        raise SagaError("--reason is required (e.g. the DSR ticket id)")
    t = rt.lineage.load_trace(trace_id)
    if t is None:
        raise SagaError(
            f"no trace {trace_id!r} in this installation; erase only accepts a trace id produced by "
            "`tombstone trace` (Hard Rule 10)"
        )
    saga = Saga(
        rt,
        t,
        SagaOptions(reason=reason, accept_gaps=accept_gaps, reclaim=reclaim, semantic=semantic),
    )
    if retry:
        saga.retry = True
    result = saga.run()
    text = render_receipt(result.receipt, result.view)
    return result.exit_code, text, result.receipt.to_dict()


@command("erase", "two-phase erasure of a traced subject (requires a trace id and --confirm)")
def _setup_erase(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
    p.add_argument("--trace", required=True, help="trace id from `tombstone trace`")
    p.add_argument("--reason", required=True, help="operator reason, e.g. dsr-2026-0912")
    p.add_argument("--confirm", action="store_true", help="explicit confirmation (required)")
    p.add_argument(
        "--accept-gaps",
        action="store_true",
        help="proceed despite lineage gaps (receipt records UNVERIFIED(lineage-gap))",
    )
    p.add_argument(
        "--no-reclaim", action="store_true", help="phase 1 only: suppress, do not reclaim"
    )
    p.add_argument("--semantic", action="store_true", help="also measure Ghost Echoes drift (slow)")

    def run(ns: argparse.Namespace) -> int:
        rt = Runtime.load(ns.config)
        try:
            code, text, data = run_erase(
                rt, ns.trace, ns.reason, ns.confirm, ns.accept_gaps, not ns.no_reclaim, ns.semantic
            )
        finally:
            rt.close()
        emit(ns, text, data)
        return code

    return run


@command("dlq", "list or retry dead-lettered erasure steps")
def _setup_dlq(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
    p.add_argument("action", choices=["list", "retry"], nargs="?", default="list")
    p.add_argument("--trace", default=None, help="retry only this trace")
    p.add_argument("--reason", default="dlq-retry", help="reason recorded on the retry saga")
    p.add_argument("--confirm", action="store_true", help="explicit confirmation for retry")

    def run(ns: argparse.Namespace) -> int:
        rt = Runtime.load(ns.config)
        try:
            journal = Journal(rt.inst.journal_path)
            entries = [r for r in journal.records() if r.type == Journal.DLQ]
            if ns.action == "list":
                if not entries:
                    emit(ns, "dlq: empty", {"entries": []})
                    return EXIT_OK
                lines = [f"dlq: {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}"]
                for r in entries:
                    b = r.body
                    lines.append(
                        f"  saga {b['saga_id']}  step {b['step_id']}  store {b['store']}  artifacts {len(b['artifact_ids'])}  retry {b['retry']}"
                    )
                    lines.append(f"      {b['error'][:120]}")
                emit(ns, "\n".join(lines), {"entries": [r.body for r in entries]})
                return EXIT_OK
            # retry: one new saga per trace that has DLQ entries
            starts = {
                r.body["saga_id"]: r.body for r in journal.records() if r.type == Journal.SAGA_START
            }
            traces = sorted(
                {
                    starts[e.body["saga_id"]]["trace_id"]
                    for e in entries
                    if e.body["saga_id"] in starts
                }
            )
            if ns.trace:
                traces = [t for t in traces if t == ns.trace]
            if not traces:
                emit(ns, "dlq: nothing to retry", {"retried": []})
                return EXIT_OK
            worst = EXIT_OK
            outputs = []
            for tid in traces:
                code, text, _data = run_erase(
                    rt, tid, ns.reason, ns.confirm, accept_gaps=True, retry=True
                )
                worst = max(worst, code)
                outputs.append(text)
                sys.stdout.flush()
            emit(ns, "\n\n".join(outputs), {"retried": traces})
            return worst
        finally:
            rt.close()

    return run


__all__ = ["EXIT_ERROR", "run_erase"]
