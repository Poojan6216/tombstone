"""``tombstone repin --reason``: acknowledge a store/model/manifest change."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from tombstone.cli import EXIT_OK, command, emit
from tombstone.pins import repin
from tombstone.registry import Runtime


@command("repin", "acknowledge a store/model/manifest change with a reason")
def _setup(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
    p.add_argument("--reason", required=True, help="why the change is expected")
    p.add_argument("--store", action="append", default=None, help="limit to these store names")

    def run(ns: argparse.Namespace) -> int:
        rt = Runtime.load(ns.config)
        try:
            result = repin(rt, ns.reason, ns.store)
        finally:
            rt.close()
        lines = []
        for name, deltas in sorted(result.items()):
            if deltas:
                lines.append(f"{name}: re-pinned ({len(deltas)} change(s))")
                lines.extend(f"    {d}" for d in deltas)
            else:
                lines.append(f"{name}: unchanged")
        emit(ns, "\n".join(lines), {k: [str(d) for d in v] for k, v in result.items()})
        return EXIT_OK

    return run
