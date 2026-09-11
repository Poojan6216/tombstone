"""The convenience surface: ``from tombstone import forget`` and ``tombstone forget``.

Both are wrappers over the same saga the two-command path runs, so what these tests care about
is that the wrapper is honest — it refuses the same things, it asks before erasing, it does not
erase when the answer is no, and the receipt it produces is a real one that ``replay``
re-derives.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests import _pipeline, _stores
from tombstone.errors import LineageGapError
from tombstone.receipt.ledger import Ledger
from tombstone.registry import Runtime


def _cli(args: list[str], cwd: Path, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tombstone", *args],
        cwd=cwd,
        env={**os.environ, "TOMBSTONE_LOG_LEVEL": "error"},
        input=stdin if stdin is not None else "",
        capture_output=True,
        text=True,
    )


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.chdir(tmp_path)
    _stores.skip_unless("faiss")
    h = _pipeline.build(tmp_path, ["faiss"], rag_subjects=("S-0001",))
    return h


def test_import_does_not_cost_anything(project: dict[str, object]) -> None:
    """`from tombstone import forget` must not drag the runtime in at import time (PEP 562)."""
    code = (
        "import sys, tombstone\n"
        "assert 'tombstone.api' not in sys.modules, 'api imported eagerly'\n"
        "from tombstone import forget, trace, Held, Erasure\n"
        "assert callable(forget) and callable(trace)\n"
        "assert 'tombstone.api' in sys.modules\n"
        "print('ok')\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.stdout.strip() == "ok", p.stderr


def test_trace_answers_what_do_we_hold(project: dict[str, object]) -> None:
    import tombstone

    held = tombstone.trace("S-0001", config=project["cfg_path"])
    assert held.count > 0
    assert sum(held.by_store.values()) == held.count
    assert sum(held.by_kind.values()) == held.count
    assert held.trace_id and held.scope == "default"
    assert "S-0001" not in held.subject and "S-0001" not in held.summary()  # Hard Rule 3
    assert str(held.count) in held.summary()
    # reading is not writing
    assert not Ledger(
        Path(str(project["cfg_path"])).parent / ".tombstone" / "ledger.jsonl"
    ).receipts()


def test_trace_of_a_stranger_refuses_to_say_nothing_to_delete(project: dict[str, object]) -> None:
    """Hard Rule 4. "We hold nothing on them" and "they arrived before we were watching" look
    identical from inside the database; the convenience wrapper must not turn that into a
    reassuring empty result."""
    import tombstone

    with pytest.raises(LineageGapError) as e:
        tombstone.trace("nobody-by-that-id", config=project["cfg_path"])
    assert "before capture was enabled" in str(e.value)
    assert "nobody-by-that-id" not in str(e.value)  # Hard Rule 3: the raw id is never echoed


def test_forget_erases_and_hands_back_a_real_receipt(project: dict[str, object]) -> None:
    import tombstone

    cfg = project["cfg_path"]
    before = tombstone.trace("S-0001", config=cfg)
    assert before.count > 0

    result = tombstone.forget("S-0001", reason="dsr-api-1", config=cfg)

    assert result.exit_code in (0, 2)
    assert result.ok == (result.exit_code == 0)
    assert result.receipt_id and result.receipt_path.is_file()
    signed = json.loads(result.receipt_path.read_text())
    assert signed["reason"] == "dsr-api-1" and signed["signature"]
    assert signed["trace_id"] == before.trace_id
    # the counts on the object are the counts in the receipt
    total = result.verified + result.unverified + result.residual
    assert total == len(signed["statuses"]) == before.count
    assert result.report and "VERIFIED" in result.report
    # and the ledger now holds exactly this receipt
    root = Path(str(cfg)).parent
    receipts = Ledger(root / ".tombstone" / "ledger.jsonl").receipts()
    assert [r.receipt_id for r in receipts] == [result.receipt_id]


def test_a_receipt_written_from_python_replays(project: dict[str, object]) -> None:
    """Hard Rule 9: however the erasure was driven, the journal must re-derive the same receipt."""
    import tombstone

    cfg = Path(str(project["cfg_path"]))
    tombstone.forget("S-0001", reason="dsr-api-replay", config=cfg)
    p = _cli(["replay", "--config", str(cfg)], cfg.parent)
    assert p.returncode == 0, p.stdout + p.stderr


def _write_behind_tombstones_back(rt: Runtime, n: int = 20) -> None:
    """Put entries in a store with no lineage for them — what an unwrapped ingest leaves."""
    from tombstone.lineage.capture import EmbedRecord

    store = next(s for s in rt.all_stores().values() if s.kind == "faiss")
    emb = _stores.embedder()
    texts = [f"pre-existing document {i} nobody stamped" for i in range(n)]
    store._add(  # type: ignore[attr-defined]
        [
            EmbedRecord(f"legacy{i}", v, {"legacy": True}, t, None, None)  # type: ignore[arg-type]
            for i, (v, t) in enumerate(zip(emb.embed(texts), texts, strict=True))
        ]
    )


def test_forget_refuses_a_subject_with_lineage_gaps(project: dict[str, object]) -> None:
    """The gap refusal is the point of the tool; the convenience wrapper must not soften it."""
    import tombstone

    cfg = Path(str(project["cfg_path"]))
    rt = Runtime.shared(cfg)
    assert not tombstone.trace("S-0001", config=cfg).gaps, "fixture should start clean"

    _write_behind_tombstones_back(rt)

    held = tombstone.trace("S-0001", config=cfg)
    assert held.gaps, "20 unlineaged entries should be reported as a gap"
    assert "lineage gaps" in held.summary()

    with pytest.raises(LineageGapError):
        tombstone.forget("S-0001", reason="dsr-gap", config=cfg)
    root = cfg.parent
    assert not Ledger(root / ".tombstone" / "ledger.jsonl").receipts(), "refusal must not erase"

    # accepting them is allowed, and is recorded rather than hidden
    result = tombstone.forget("S-0001", reason="dsr-gap", config=cfg, accept_gaps=True)
    assert result.receipt_path.is_file()
    signed = json.loads(result.receipt_path.read_text())
    assert any("gap" in n.lower() for n in signed["notes"]), signed["notes"]


def test_cli_forget_refuses_gaps_before_asking(project: dict[str, object]) -> None:
    """Being told the erasure is impossible only after approving it is the wrong order."""
    cfg = Path(str(project["cfg_path"]))
    _write_behind_tombstones_back(Runtime.shared(cfg))
    p = _cli(["forget", "S-0001", "--reason", "r", "--config", str(cfg), "--yes"], cfg.parent)
    assert p.returncode != 0
    assert "lineage gap" in p.stderr and "--accept-gaps" in p.stderr
    assert not Ledger(cfg.parent / ".tombstone" / "ledger.jsonl").receipts()


def test_cli_forget_asks_first_and_a_no_changes_nothing(project: dict[str, object]) -> None:
    cfg = Path(str(project["cfg_path"]))
    root = cfg.parent
    p = _cli(
        ["forget", "S-0001", "--reason", "dsr-cli-no", "--config", str(cfg)], root, stdin="n\n"
    )
    # stdin is a pipe, not a terminal: it must refuse rather than silently erase
    assert p.returncode != 0
    assert "not a terminal" in p.stderr and "Nothing was changed" in p.stderr
    assert not Ledger(root / ".tombstone" / "ledger.jsonl").receipts()
    # ...and it still showed the operator what it found before refusing
    assert "exist because of subject" in p.stdout


def test_cli_forget_with_yes_erases_in_one_command(project: dict[str, object]) -> None:
    cfg = Path(str(project["cfg_path"]))
    root = cfg.parent
    p = _cli(["forget", "S-0001", "--reason", "dsr-cli-yes", "--config", str(cfg), "--yes"], root)
    assert p.returncode in (0, 2), p.stderr
    assert "exist because of subject" in p.stdout  # the preview
    assert "VERIFIED" in p.stdout  # the receipt
    receipts = Ledger(root / ".tombstone" / "ledger.jsonl").receipts()
    assert len(receipts) == 1 and receipts[0].reason == "dsr-cli-yes"


def test_cli_forget_json_needs_yes_because_it_cannot_prompt(project: dict[str, object]) -> None:
    cfg = Path(str(project["cfg_path"]))
    root = cfg.parent
    p = _cli(["forget", "S-0001", "--reason", "r", "--config", str(cfg), "--json"], root)
    assert p.returncode != 0 and "Nothing was changed" in p.stderr
    assert not Ledger(root / ".tombstone" / "ledger.jsonl").receipts()

    ok = _cli(["forget", "S-0001", "--reason", "r", "--config", str(cfg), "--json", "--yes"], root)
    assert ok.returncode in (0, 2), ok.stderr
    payload = json.loads(ok.stdout)  # stdout stays pure JSON: no preview leaked into it
    assert payload["erased"] is True and payload["receipt_id"]


def test_cli_forget_on_a_stranger_explains_instead_of_reassuring(
    project: dict[str, object],
) -> None:
    cfg = Path(str(project["cfg_path"]))
    root = cfg.parent
    p = _cli(["forget", "nobody", "--reason", "r", "--config", str(cfg), "--yes"], root)
    assert p.returncode != 0
    assert "no lineage records" in p.stderr and "before capture was enabled" in p.stderr
    assert not Ledger(root / ".tombstone" / "ledger.jsonl").receipts()


def test_forget_is_the_same_erasure_as_the_two_command_path(project: dict[str, object]) -> None:
    """One command must not be a different, weaker erasure: same statuses, same outcomes."""
    import tombstone
    from tombstone.commands.trace import run_trace

    cfg = Path(str(project["cfg_path"]))
    rt = Runtime.shared(cfg)
    t, _ = run_trace(rt, "S-0001")
    expected = {a.artifact_id for a in t.artifacts}

    result = tombstone.forget("S-0001", reason="dsr-same", config=cfg)
    signed = json.loads(result.receipt_path.read_text())
    got = {s["artifact"]["artifact_id"] for s in signed["statuses"]}
    assert got == expected


def _cli_on_a_terminal(args: list[str], cwd: Path, typed: str) -> tuple[int, str]:
    """Run the CLI with a real pty on stdin, so ``isatty()`` is true and the prompt appears."""
    import pty
    import select

    pid, fd = pty.fork()
    if pid == 0:  # child: exec the CLI with the pty as its stdio
        os.chdir(cwd)
        os.environ["TOMBSTONE_LOG_LEVEL"] = "error"
        os.execv(sys.executable, [sys.executable, "-m", "tombstone", *args])
    out = b""
    typed_yet = False
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        r, _, _ = select.select([fd], [], [], 1.0)
        if r:
            try:
                chunk = os.read(fd, 65536)
            except OSError:  # the child closed the pty
                break
            if not chunk:
                break
            out += chunk
            if not typed_yet and b"[y/N]" in out:
                os.write(fd, typed.encode())
                typed_yet = True
        if not r and typed_yet and b"receipt" in out:
            break
    _, status = os.waitpid(pid, 0)
    os.close(fd)
    assert typed_yet, f"no prompt appeared:\n{out.decode(errors='replace')}"
    return os.waitstatus_to_exitcode(status), out.decode(errors="replace")


@pytest.mark.parametrize(
    ("typed", "should_erase"), [("y\n", True), ("n\n", False), ("\n", False), ("Y\n", True)]
)
def test_cli_forget_prompt_erases_only_on_an_explicit_yes(
    project: dict[str, object], typed: str, should_erase: bool
) -> None:
    """The prompt is the confirmation, so only a typed yes may erase — a bare return must not."""
    cfg = Path(str(project["cfg_path"]))
    root = cfg.parent
    code, out = _cli_on_a_terminal(
        ["forget", "S-0001", "--reason", "dsr-tty", "--config", str(cfg)], root, typed
    )
    assert "exist because of subject" in out
    receipts = Ledger(root / ".tombstone" / "ledger.jsonl").receipts()
    if should_erase:
        assert code in (0, 2), out
        assert len(receipts) == 1 and receipts[0].reason == "dsr-tty"
    else:
        assert code != 0
        assert "aborted" in out and "Nothing was changed" in out
        assert not receipts
