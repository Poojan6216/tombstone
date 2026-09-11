"""``tombstone ui``: the local page, for the people who handle deletion requests but do not
live in a terminal."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from tombstone.cli import command


@command("ui", "serve a local page: search a person, see what is held, erase it, read the receipt")
def _setup_ui(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
    p.add_argument("--port", type=int, default=7878, help="loopback port (default 7878)")
    p.add_argument("--no-open", action="store_true", help="do not open a browser")

    def run(ns: argparse.Namespace) -> int:
        from tombstone.ui import serve

        return serve(ns.config, port=ns.port, open_browser=not ns.no_open)

    return run
