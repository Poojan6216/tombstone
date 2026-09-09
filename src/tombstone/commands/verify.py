"""``tombstone verify``: audit a subject (Demo 1) or independently re-check a receipt (5.6)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tombstone.cli import EXIT_OK, EXIT_UNVERIFIED, command, emit
from tombstone.commands.trace import run_trace
from tombstone.errors import SignatureInvalid, TombstoneError
from tombstone.registry import Runtime


@command("verify", "probe every layer for a subject, or re-check a receipt independently")
def _setup(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
    p.add_argument("--subject", default=None, help="raw subject id to audit (hashed immediately)")
    p.add_argument(
        "--after-native-delete",
        action="store_true",
        help="label the audit as run after the app's own delete()",
    )
    p.add_argument("--receipt", default=None, help="receipt JSON path to verify independently")
    p.add_argument(
        "--public-key",
        default=None,
        help="PEM public key for --receipt (default: the receipt's embedded key)",
    )
    p.add_argument(
        "--ledger", default=None, help="optional ledger.jsonl to check the chain for --receipt"
    )
    p.add_argument("--no-store-scan", action="store_true")

    def run(ns: argparse.Namespace) -> int:
        if ns.receipt:
            return _verify_receipt(ns)
        if not ns.subject:
            raise TombstoneError("verify needs --subject <id> or --receipt <path>")
        from tombstone.verify.audit import audit_trace, render_audit_report

        rt = Runtime.load(ns.config)
        try:
            t, _ = run_trace(rt, ns.subject, with_store_gaps=not ns.no_store_scan)
            report = audit_trace(rt, t)
        finally:
            rt.close()
        text = render_audit_report(report)
        if ns.after_native_delete:
            text = "after native delete()\n" + text
        emit(ns, text, report.to_dict())
        return EXIT_UNVERIFIED if report.recoverable else EXIT_OK

    return run


def _verify_receipt(ns: argparse.Namespace) -> int:
    from tombstone.verify.independent import verify_receipt_independently

    result = verify_receipt_independently(
        Path(ns.receipt),
        public_key_pem=Path(ns.public_key) if ns.public_key else None,
        ledger=Path(ns.ledger) if ns.ledger else None,
        config=ns.config,
    )
    emit(ns, result["text"], result)
    return EXIT_OK if result["ok"] else EXIT_UNVERIFIED


__all__ = ["Any", "SignatureInvalid", "json"]
