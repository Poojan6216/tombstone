"""``tombstone stamp``: stamp JSON documents from a file or stdin, write JSON to stdout.

Input: a JSON object, a JSON list, or JSON Lines; each item has ``text`` (optional) and
``metadata`` (optional). Output mirrors the input with stamped metadata. The raw subject id is
hashed with the installation pepper before anything is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from tombstone.cli import EXIT_OK, command
from tombstone.lineage.stamp import stamp
from tombstone.registry import Runtime


def _read_items(path: str | None) -> list[dict[str, Any]]:
    raw = sys.stdin.read() if path in (None, "-") else open(path, encoding="utf-8").read()  # noqa: SIM115
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        return [dict(d) for d in data]
    if raw.startswith("{") and "\n{" not in raw:
        return [dict(json.loads(raw))]
    return [dict(json.loads(line)) for line in raw.splitlines() if line.strip()]


@command("stamp", "stamp documents with subject/source/scope metadata")
def _setup(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
    p.add_argument("--subject", required=True, help="raw subject id (hashed immediately)")
    p.add_argument("--source", required=True, help="source id, e.g. a file path (hashed)")
    p.add_argument("--scope", default=None, help="tenant scope (default: config scope)")
    p.add_argument(
        "--mention", action="append", default=[], help="other subject ids this document names"
    )
    p.add_argument(
        "--derived-from", default=None, help="artifact id this document was derived from"
    )
    p.add_argument("--input", "-i", default="-", help="JSON / JSON Lines file (default stdin)")
    p.add_argument("--jsonl", action="store_true", help="emit JSON Lines instead of a JSON list")

    def run(ns: argparse.Namespace) -> int:
        rt = Runtime.load(ns.config)
        pepper = rt.pepper()
        scope = ns.scope or rt.cfg.scope
        items = _read_items(ns.input)
        out: list[dict[str, Any]] = []
        for it in items:
            md = dict(it.get("metadata") or {})
            stamped = stamp(
                md,
                ns.subject,
                ns.source,
                scope,
                pepper=pepper,
                mentions=list(ns.mention),
                derived_from=ns.derived_from,
            )
            new = dict(it)
            new["metadata"] = stamped
            out.append(new)
        if ns.jsonl:
            for o in out:
                sys.stdout.write(json.dumps(o, ensure_ascii=False) + "\n")
        else:
            sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
        sys.stdout.flush()
        rt.close()
        return EXIT_OK

    return run
