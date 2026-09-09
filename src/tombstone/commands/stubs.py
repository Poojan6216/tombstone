"""Commands that later phases implement. Each is registered so `tombstone <cmd> --help` works.

A stub exits 1 with a message naming the phase that implements it. Nothing here is silent.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from tombstone.cli import EXIT_ERROR, command


def _stub(name: str, phase: str) -> Callable[[argparse.Namespace], int]:
    def run(_ns: argparse.Namespace) -> int:
        sys.stderr.write(f"error: `tombstone {name}` is not implemented yet ({phase})\n")
        return EXIT_ERROR

    return run


def _register(name: str, help_: str, phase: str) -> None:
    @command(name, help_)
    def _setup(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
        p.add_argument("args", nargs=argparse.REMAINDER)
        return _stub(name, phase)


for _name, _help, _phase in (
    ("trace", "list every artifact descending from a subject", "Phase 2"),
    (
        "erase",
        "two-phase erasure of a traced subject (requires a trace id and --confirm)",
        "Phase 3",
    ),
    ("verify", "probe every layer for a subject or re-check a receipt independently", "Phase 5"),
    ("receipt", "show a receipt", "Phase 5"),
    ("replay", "re-derive every receipt from the journal and assert equality", "Phase 5"),
    ("status", "lineage coverage per store, pins, DLQ depth", "Phase 2"),
    ("dlq", "list or retry dead-lettered erasure steps", "Phase 3"),
    ("mcp", "run the MCP server (stdio or streamable HTTP)", "Phase 5"),
    ("bench", "run the residue / unlearning benchmarks", "Phase 6"),
    ("repin", "acknowledge a store/model/manifest change with a reason", "Phase 2"),
):
    _register(_name, _help, _phase)
