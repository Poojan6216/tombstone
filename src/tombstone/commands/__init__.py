"""Subcommand implementations. Importing this package registers every command with the CLI."""

from __future__ import annotations

from tombstone.commands import erase as _erase  # noqa: F401
from tombstone.commands import forget as _forget  # noqa: F401
from tombstone.commands import init as _init  # noqa: F401
from tombstone.commands import mcp as _mcp  # noqa: F401
from tombstone.commands import receipt as _receipt  # noqa: F401
from tombstone.commands import repin as _repin  # noqa: F401
from tombstone.commands import scan as _scan  # noqa: F401
from tombstone.commands import stamp as _stamp  # noqa: F401
from tombstone.commands import stubs as _stubs  # noqa: F401
from tombstone.commands import trace as _trace  # noqa: F401
from tombstone.commands import ui as _ui  # noqa: F401
from tombstone.commands import verify as _verify  # noqa: F401
