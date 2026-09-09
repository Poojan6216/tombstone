"""``tombstone mcp``: run the MCP server (stdio by default, or streamable HTTP)."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from tombstone.cli import EXIT_OK, command


@command("mcp", "run the MCP server (stdio or streamable HTTP)")
def _setup(p: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
    p.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)

    def run(ns: argparse.Namespace) -> int:
        from tombstone.mcp.server import serve

        serve(ns.config, ns.transport, ns.host, ns.port)
        return EXIT_OK

    return run
