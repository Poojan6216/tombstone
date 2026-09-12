"""``tombstone scan``: look at a store this tool has never touched, and change nothing."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from tombstone.cli import EXIT_OK, EXIT_UNVERIFIED, command, emit


@command("scan", "look at existing stores read-only: what could a deletion request establish?")
def _setup_scan(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
    p.add_argument(
        "path",
        nargs="?",
        default=None,
        help="directory to look in (default: here). No config or lineage required.",
    )
    p.add_argument("--dsn", default=None, help="also scan this Postgres/pgvector database")
    p.add_argument("--table", default="documents", help="pgvector table name (default documents)")
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="entries to sample per store (default 200); larger is slower and more precise",
    )

    def run(ns: argparse.Namespace) -> int:
        from tombstone.scan import DEFAULT_SAMPLE, render, scan

        report = scan(
            root=Path(ns.path) if ns.path else None,
            config=ns.config,
            dsn=ns.dsn,
            table=ns.table,
            sample=ns.sample or DEFAULT_SAMPLE,
        )
        emit(ns, render(report, colour=sys.stdout.isatty()), report.to_dict())
        # Exit 2 — the same code the rest of the tool uses for "checked, and the answer is not a
        # clean pass" — when nothing here can be traced. It is not an error: it is the finding.
        return EXIT_OK if report.any_traceable else EXIT_UNVERIFIED

    return run
