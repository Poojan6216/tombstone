"""Subcommand implementations. Importing this package registers every command with the CLI."""

from __future__ import annotations

from tombstone.commands import init as _init  # noqa: F401
from tombstone.commands import stamp as _stamp  # noqa: F401
from tombstone.commands import stubs as _stubs  # noqa: F401
