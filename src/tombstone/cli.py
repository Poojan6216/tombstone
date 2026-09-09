"""``tombstone`` command line interface.

Subcommands: init stamp trace erase verify receipt replay status dlq mcp bench repin.
Human-readable output goes to stdout; structured logs go to stderr (never stdout — stdout is the
MCP stdio transport in ``tombstone mcp``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from tombstone import __version__
from tombstone.errors import TombstoneError
from tombstone.logging import get_logger

Handler = Callable[[argparse.Namespace], int]

_COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None], Handler]] = {}

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_UNVERIFIED = 2


def command(
    name: str, help_: str
) -> Callable[[Callable[[argparse.ArgumentParser], Handler]], None]:
    """Register a subcommand: the decorated function configures the parser and returns a handler."""

    def deco(setup: Callable[[argparse.ArgumentParser], Handler]) -> None:
        holder: dict[str, Handler] = {}

        def configure(p: argparse.ArgumentParser) -> None:
            holder["h"] = setup(p)

        def run(ns: argparse.Namespace) -> int:
            return holder["h"](ns)

        if name in _COMMANDS:
            raise RuntimeError(f"subcommand {name!r} registered twice")
        _COMMANDS[name] = (help_, configure, run)

    return deco


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", default=None, help="path to tombstone.yaml")
    p.add_argument("--json", action="store_true", help="machine-readable output on stdout")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tombstone",
        description=(
            "Track where a data subject's data went, erase it everywhere, and get a receipt that "
            "says what was checked and what was not."
        ),
    )
    parser.add_argument("--version", action="version", version=f"tombstone {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True
    for name, (help_, configure, _run) in _COMMANDS.items():
        p = sub.add_parser(name, help=help_, description=help_)
        _add_common(p)
        configure(p)
    return parser


def emit(ns: argparse.Namespace, human: str, data: dict[str, Any] | None = None) -> None:
    """Print human text, or JSON when ``--json`` was passed."""
    if getattr(ns, "json", False) and data is not None:
        sys.stdout.write(json.dumps(data, sort_keys=True, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(human if human.endswith("\n") else human + "\n")
    sys.stdout.flush()


def main(argv: Sequence[str] | None = None) -> int:
    import tombstone.commands  # noqa: F401  (registers subcommands)

    log = get_logger("cli")
    parser = build_parser()
    ns = parser.parse_args(argv)
    _help, _configure, run = _COMMANDS[ns.command]
    try:
        return run(ns)
    except TombstoneError as e:
        sys.stderr.write(f"error: {e}\n")
        log.debug("command failed", extra={"tombstone_extra": {"command": ns.command}})
        return e.exit_code
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130


def cwd_state(ns: argparse.Namespace) -> Path:
    cfg_path = getattr(ns, "config", None)
    return Path(cfg_path).parent if cfg_path else Path.cwd()


if __name__ == "__main__":  # pragma: no cover
    # Import through the package so subcommands register on the same module object.
    from tombstone.cli import main as _main

    raise SystemExit(_main())
