from __future__ import annotations

import subprocess
import sys

import pytest

from tombstone.cli import build_parser, main

EXPECTED = {
    "init",
    "stamp",
    "trace",
    "erase",
    "verify",
    "receipt",
    "replay",
    "status",
    "dlq",
    "mcp",
    "bench",
    "repin",
}


def test_all_subcommands_present() -> None:
    import tombstone.commands  # noqa: F401
    from tombstone.cli import _COMMANDS

    assert EXPECTED <= set(_COMMANDS)
    parser = build_parser()
    assert parser.prog == "tombstone"


@pytest.mark.parametrize("cmd", sorted(EXPECTED - {"init"}))
def test_stubs_fail_loud_not_silent(cmd: str, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([cmd])
    err = capsys.readouterr()
    # Either implemented (may fail for lack of config → non-zero) or an explicit "not implemented".
    assert rc != 0 or err.out
    assert "error:" in err.err or rc == 0


def test_version() -> None:
    with pytest.raises(SystemExit) as ei:
        main(["--version"])
    assert ei.value.code == 0


def test_logs_go_to_stderr_never_stdout() -> None:
    code = (
        "import logging, sys\n"
        "from tombstone.logging import get_logger, log\n"
        "lg = get_logger('t'); lg.setLevel(logging.DEBUG)\n"
        "log(lg, logging.INFO, 'hello', artifact_id='01J')\n"
        "sys.stdout.write('STDOUT-ONLY\\n')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert proc.stdout == "STDOUT-ONLY\n"
    assert '"msg": "hello"' in proc.stderr and '"artifact_id": "01J"' in proc.stderr
