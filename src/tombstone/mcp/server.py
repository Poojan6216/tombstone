"""The MCP server: tools ``tombstone.forget``, ``tombstone.trace``, ``tombstone.verify``,
``tombstone.erase``, ``tombstone.receipt``, ``tombstone.status``. stdio and streamable HTTP
transports.

``forget`` is the one an assistant should reach for: subject in, receipt out. ``erase`` is the
same thing split across two calls, for a client that already holds a trace id.

Both are destructive. Confirmation rides the SDK's resolver mechanism (see ``_tools.py``):

* on protocol ``2026-07-28`` the first call returns ``InputRequiredResult`` (``resultType:
  input_required``) carrying the trace summary and a sealed, expiring, request-bound
  ``requestState``; the erase runs only when the client retries with an accepted answer;
* on ``2025-11-25`` and earlier the same question is sent as a standalone elicitation mid-call;
* a client that declares no form-elicitation capability gets an error naming the CLI command to
  run instead, before anything runs.

``forget`` erases exactly the trace its confirmation described, rather than tracing a second time
after the answer comes back and erasing whatever that finds.

It never degrades to executing without confirmation. Subject ids arrive raw and are hashed
before touching disk. Tool results never contain content (Hard Rule 7). Nothing is written to
stdout except JSON-RPC frames (logs go to stderr). The request-state key is derived from the
installation pepper, so a state minted by another installation is rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tombstone.errors import ConfigError


def build_server(config: str | Path | None = None) -> Any:
    try:
        from tombstone.mcp import _tools
    except ImportError as e:  # pragma: no cover
        raise ConfigError(
            "the MCP server needs the [mcp] extra: uv pip install 'tombstone-erase[mcp]'"
        ) from e
    return _tools.build(config)


def serve(
    config: str | Path | None, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765
) -> None:
    server = build_server(config)
    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.settings.host = host
        server.settings.port = port
        server.run(transport="streamable-http")
