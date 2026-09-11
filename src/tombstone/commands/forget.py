"""``tombstone forget``: the whole erasure in one command.

``trace`` then ``erase`` is two commands with a 26-character trace id copied between them, which
is fine for a script and wrong for a person handling a deletion request. This command traces the
subject, shows what it found, asks once, and erases — and the confirmation is a human typing
"yes" to a list of real artifacts rather than a ``--confirm`` flag pasted without reading.

It is the same saga underneath: same journal, same receipt, same exit codes, and
``tombstone replay`` re-derives a receipt written this way exactly as it does any other.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from tombstone.api import erase_traced, look
from tombstone.cli import EXIT_ERROR, command, emit
from tombstone.errors import LineageGapError
from tombstone.registry import Runtime

_PROMPT = "erase all {n} of these? this cannot be undone  [y/N] "


def _ask(prompt: str) -> bool:
    """True only for an explicit yes. EOF, an empty line, anything else: no."""
    try:
        answer = input(prompt)
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


@command("forget", "trace a subject and erase everything they left behind, in one command")
def _setup_forget(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
    p.add_argument("subject", help="the subject id your app knows (hashed before it is stored)")
    p.add_argument("--reason", required=True, help="goes on the receipt, e.g. the DSR ticket id")
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="skip the confirmation prompt (required when stdin is not a terminal)",
    )
    p.add_argument(
        "--accept-gaps",
        action="store_true",
        help="proceed despite lineage gaps (receipt records UNVERIFIED(lineage-gap))",
    )
    p.add_argument(
        "--no-reclaim", action="store_true", help="phase 1 only: suppress, do not reclaim"
    )
    p.add_argument("--semantic", action="store_true", help="also measure Ghost Echoes drift (slow)")
    p.add_argument(
        "--no-store-scan", action="store_true", help="skip sampling stores for unlineaged entries"
    )

    def run(ns: argparse.Namespace) -> int:
        as_json = bool(getattr(ns, "json", False))
        rt = Runtime.load(ns.config)
        try:
            # A subject with no lineage at all does not reach here: trace raises rather than
            # answer "nothing to delete", because it cannot tell that apart from "ingested
            # before capture was on" (Hard Rule 4). The error says so and names both.
            held, t = look(rt, ns.subject, scan_stores=not ns.no_store_scan)
            if not as_json:
                sys.stdout.write(held.summary() + "\n\n")
                sys.stdout.flush()
            if held.gaps and not ns.accept_gaps:
                # Refuse before asking, not after: being told the erasure is impossible only
                # once you have already approved it is the wrong order.
                raise LineageGapError(
                    f"subject {held.subject} has {len(held.gaps)} lineage gap(s); forget refuses "
                    "unless --accept-gaps is passed (then the receipt records "
                    "UNVERIFIED(lineage-gap) for the named stores). Nothing was changed."
                )
            if not ns.yes:
                if as_json:
                    sys.stderr.write(
                        "error: --json cannot prompt for confirmation; pass --yes if you have "
                        "already shown the operator what will be erased. Nothing was changed.\n"
                    )
                    return EXIT_ERROR
                if not sys.stdin.isatty():
                    sys.stderr.write(
                        "error: stdin is not a terminal, so nobody can answer the confirmation. "
                        "Pass --yes to erase non-interactively. Nothing was changed.\n"
                    )
                    return EXIT_ERROR
                if not _ask(_PROMPT.format(n=held.count)):
                    sys.stdout.write("aborted. Nothing was changed.\n")
                    return EXIT_ERROR
                sys.stdout.write("\n")
                sys.stdout.flush()
            result = erase_traced(
                rt,
                t,
                ns.reason,
                accept_gaps=ns.accept_gaps,
                reclaim=not ns.no_reclaim,
                semantic=ns.semantic,
            )
        finally:
            rt.close()
        emit(ns, result.report, {"erased": True, **result.to_dict()})
        return result.exit_code

    return run
