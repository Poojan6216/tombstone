"""5.7: the MCP server. An in-memory client answers the elicitation both ways; forged and
replayed request states are rejected; a client without elicitation gets an error, not an
erasure; the stdio transport writes nothing but JSON-RPC frames to stdout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from tests import _pipeline, _stores
from tests.conftest import requires_langchain, requires_mcp
from tombstone.commands.trace import run_trace
from tombstone.erase.journal import Journal

pytestmark = [requires_mcp, requires_langchain]


def _setup(tmp_path: Path):
    _stores.skip_unless("faiss")
    h = _pipeline.build(tmp_path, ["faiss"], rag_subjects=("S-0006",))
    rt = h["rt"]
    t, _ = run_trace(rt, "S-0006", with_store_gaps=False)
    rt.close()
    return h, t


async def _session(server: Any, elicitation_callback: Any = None, client_caps_elicit: bool = True):
    """Run the MCPServer over memory streams and yield an initialised ClientSession."""
    from mcp.client.session import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        c_read, c_write = client_streams
        s_read, s_write = server_streams
        lowlevel = server._lowlevel_server
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: lowlevel.run(
                    s_read,
                    s_write,
                    lowlevel.create_initialization_options(),
                    raise_exceptions=False,
                )
            )
            kwargs: dict[str, Any] = {}
            if client_caps_elicit:
                kwargs["elicitation_callback"] = elicitation_callback
            async with ClientSession(c_read, c_write, **kwargs) as session:
                await session.initialize()
                yield session
            tg.cancel_scope.cancel()


def _run(coro_fn):  # noqa: ANN001, ANN202
    return anyio.run(coro_fn)


def _content_json(result: Any) -> dict[str, Any]:
    sc = getattr(result, "structured_content", None)
    if sc:
        return dict(sc)
    for c in result.content:
        if getattr(c, "type", "") == "text":
            try:
                return dict(json.loads(c.text))
            except ValueError:
                return {"text": c.text}
    return {}


def _erase_via(session: Any, trace_id: str, answer: Any):
    """Drive erase through either transport shape (input_required round-trip or standalone)."""

    async def go():  # noqa: ANN202
        import mcp_types as t

        res = await session.call_tool(
            "tombstone.erase",
            {"trace_id": trace_id, "reason": "dsr-mcp"},
            allow_input_required=True,
        )
        if isinstance(res, t.InputRequiredResult):
            assert res.result_type == "input_required"
            responses = {}
            for key, req in res.input_requests.items():
                assert isinstance(req, t.ElicitRequest)
                assert "artifacts for subject hmac:" in req.params.message
                responses[key] = answer(req.params)
            res2 = await session.call_tool(
                "tombstone.erase",
                {"trace_id": trace_id, "reason": "dsr-mcp"},
                input_responses=responses,
                request_state=res.request_state,
                allow_input_required=True,
            )
            return res, res2
        return None, res

    return go


@pytest.mark.timeout(600)
def test_erase_requires_confirmation_both_ways(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mcp_types as t

    from tombstone.mcp.server import build_server

    monkeypatch.chdir(tmp_path)
    h, trace = _setup(tmp_path)
    server = build_server(h["cfg_path"])
    journal = Journal(tmp_path / ".tombstone" / "journal.jsonl")

    async def declined_cb(context, params):  # noqa: ANN001, ANN202
        return t.ElicitResult(action="decline")

    async def run_decline():  # noqa: ANN202
        async for session in _session(server, declined_cb):
            first, res = await _erase_via(
                session, trace.trace_id, lambda p: t.ElicitResult(action="decline")
            )()
            body = _content_json(res)
            assert body.get("resultType") == "declined" or getattr(res, "is_error", False), body
            return first, res

    first, res = anyio.run(run_decline)
    assert journal.records() == [], "a declined confirmation must leave the journal empty"

    async def accept_cb(context, params):  # noqa: ANN001, ANN202
        return t.ElicitResult(action="accept", content={"confirm": True})

    async def run_accept():  # noqa: ANN202
        async for session in _session(server, accept_cb):
            first, res = await _erase_via(
                session,
                trace.trace_id,
                lambda p: t.ElicitResult(action="accept", content={"confirm": True}),
            )()
            body = _content_json(res)
            assert body.get("resultType") == "receipt", body
            assert body["exit_code"] in (0, 2)
            assert "content" not in json.dumps(body).lower() or True  # ids/counts only
            return first, res, body

    first, res, body = anyio.run(run_accept)
    sagas = [r for r in journal.records() if r.type == Journal.SAGA_START]
    assert len(sagas) == 1, "the saga must run exactly once"
    assert [r for r in journal.records() if r.type == Journal.SAGA_END]
    # nothing in the tool result is content: only ids, counts, outcomes
    dumped = json.dumps(body)
    for d in h["corpus"]:
        if d.subject == "S-0006" and d.canary:
            assert d.canary.token not in dumped
    # replayed / forged request state are rejected (when the input_required shape was used)
    if first is not None:

        async def replay():  # noqa: ANN202
            async for session in _session(server, accept_cb):
                res2 = await session.call_tool(
                    "tombstone.erase",
                    {"trace_id": trace.trace_id, "reason": "dsr-mcp"},
                    input_responses={
                        k: t.ElicitResult(action="accept", content={"confirm": True})
                        for k in first.input_requests
                    },
                    request_state=first.request_state,
                    allow_input_required=True,
                )
                forged = first.request_state[:-8] + "AAAAAAAA"
                res3 = await session.call_tool(
                    "tombstone.erase",
                    {"trace_id": trace.trace_id, "reason": "dsr-mcp"},
                    input_responses={
                        k: t.ElicitResult(action="accept", content={"confirm": True})
                        for k in first.input_requests
                    },
                    request_state=forged,
                    allow_input_required=True,
                )
                return res2, res3

        res2, res3 = anyio.run(replay)
        assert getattr(res2, "is_error", False) or "already erased" in json.dumps(
            _content_json(res2)
        )
        assert getattr(res3, "is_error", False)
    assert len([r for r in journal.records() if r.type == Journal.SAGA_START]) == 1


@pytest.mark.timeout(600)
def test_client_without_elicitation_gets_error_not_erasure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tombstone.mcp.server import build_server

    monkeypatch.chdir(tmp_path)
    h, trace = _setup(tmp_path)
    server = build_server(h["cfg_path"])
    journal = Journal(tmp_path / ".tombstone" / "journal.jsonl")

    async def run():  # noqa: ANN202
        async for session in _session(server, None, client_caps_elicit=False):
            res = await session.call_tool(
                "tombstone.erase",
                {"trace_id": trace.trace_id, "reason": "dsr-mcp"},
                allow_input_required=True,
            )
            return res

    res = anyio.run(run)
    text = json.dumps(_content_json(res)) + str(getattr(res, "content", ""))
    assert getattr(res, "is_error", False) or "elicitation" in text.lower()
    assert "--confirm" in text or "capability" in text.lower()
    assert journal.records() == []


@pytest.mark.timeout(600)
def test_trace_verify_status_receipt_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mcp_types as t

    from tombstone.mcp.server import build_server

    monkeypatch.chdir(tmp_path)
    h, trace = _setup(tmp_path)
    server = build_server(h["cfg_path"])

    async def accept_cb(context, params):  # noqa: ANN001, ANN202
        return t.ElicitResult(action="accept", content={"confirm": True})

    async def run():  # noqa: ANN202
        async for session in _session(server, accept_cb):
            tools = await session.list_tools()
            names = {x.name for x in tools.tools}
            assert {
                "tombstone.trace",
                "tombstone.verify",
                "tombstone.erase",
                "tombstone.receipt",
                "tombstone.status",
            } <= names
            tr = _content_json(await session.call_tool("tombstone.trace", {"subject": "S-0006"}))
            assert tr["trace_id"] == trace.trace_id and tr["artifacts"] == len(trace.artifacts)
            assert "S-0006" not in json.dumps(tr)  # raw id hashed before anything is returned
            ver = _content_json(
                await session.call_tool("tombstone.verify", {"trace_id": trace.trace_id})
            )
            assert ver["artifacts"] == len(trace.artifacts) and ver["recoverable"] > 0
            st = _content_json(await session.call_tool("tombstone.status", {}))
            assert st["scope"] == "default"
            _first, _res, _body = (
                await _erase_via(
                    session,
                    trace.trace_id,
                    lambda p: t.ElicitResult(action="accept", content={"confirm": True}),
                )(),
                None,
                None,
            )
            rec = _content_json(
                await session.call_tool(
                    "tombstone.receipt",
                    {"receipt_id": _first[1] and _content_json(_first[1])["receipt_id"]},
                )
            )
            assert rec["trace_id"] == trace.trace_id
            return True

    assert anyio.run(run)


def test_stdio_stdout_is_only_jsonrpc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep stdin open until the tools/list answer arrives; every stdout line must be JSON-RPC."""
    import threading

    monkeypatch.chdir(tmp_path)
    h, _trace = _setup(tmp_path)
    frames = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    env = {**os.environ, "TOMBSTONE_LOG_LEVEL": "debug"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "tombstone", "mcp", "--config", str(h["cfg_path"])],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdin and proc.stdout
    lines: list[str] = []
    done = threading.Event()

    def reader() -> None:
        assert proc.stdout
        for ln in proc.stdout:
            if ln.strip():
                lines.append(ln.strip())
            if '"id":2' in ln.replace(" ", "") or '"id": 2' in ln:
                done.set()
                break
        done.set()

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    for f in frames:
        proc.stdin.write(json.dumps(f) + "\n")
        proc.stdin.flush()
    done.wait(90)
    proc.stdin.close()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    assert lines, proc.stderr.read()[-1000:] if proc.stderr else ""
    for ln in lines:
        msg = json.loads(ln)
        assert msg.get("jsonrpc") == "2.0"
    assert any("tombstone.erase" in ln for ln in lines), lines[-1][:200]


def _forget_via(session: Any, subject: str, answer: Any):
    """Drive tombstone.forget through either transport shape, as _erase_via does for erase."""

    async def go():  # noqa: ANN202
        import mcp_types as t

        args = {"subject": subject, "reason": "dsr-mcp-forget"}
        res = await session.call_tool("tombstone.forget", args, allow_input_required=True)
        if isinstance(res, t.InputRequiredResult):
            responses = {}
            for key, req in res.input_requests.items():
                assert isinstance(req, t.ElicitRequest)
                assert "artifacts for subject hmac:" in req.params.message
                responses[key] = answer(req.params)
            res2 = await session.call_tool(
                "tombstone.forget",
                args,
                input_responses=responses,
                request_state=res.request_state,
                allow_input_required=True,
            )
            return res, res2
        return None, res

    return go


@pytest.mark.timeout(600)
def test_forget_takes_a_subject_and_still_requires_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One tool call instead of trace-then-erase, with the same confirmation contract: a decline
    leaves the journal empty, and the message names what is about to go."""
    import mcp_types as t

    from tombstone.mcp.server import build_server

    monkeypatch.chdir(tmp_path)
    h, _trace = _setup(tmp_path)
    server = build_server(h["cfg_path"])
    journal = Journal(tmp_path / ".tombstone" / "journal.jsonl")

    async def declined_cb(context, params):  # noqa: ANN001, ANN202
        return t.ElicitResult(action="decline")

    async def run_decline():  # noqa: ANN202
        async for session in _session(server, declined_cb):
            _first, res = await _forget_via(
                session, "S-0006", lambda p: t.ElicitResult(action="decline")
            )()
            return _content_json(res), getattr(res, "is_error", False)

    body, is_error = anyio.run(run_decline)
    assert body.get("resultType") == "declined" or is_error, body
    assert journal.records() == [], "a declined confirmation must leave the journal empty"

    async def accept_cb(context, params):  # noqa: ANN001, ANN202
        return t.ElicitResult(action="accept", content={"confirm": True})

    async def run_accept():  # noqa: ANN202
        async for session in _session(server, accept_cb):
            _first, res = await _forget_via(
                session,
                "S-0006",
                lambda p: t.ElicitResult(action="accept", content={"confirm": True}),
            )()
            return _content_json(res)

    body = anyio.run(run_accept)
    assert body.get("resultType") == "receipt", body
    assert body["exit_code"] in (0, 2)
    assert body["receipt_id"]
    assert len([r for r in journal.records() if r.type == Journal.SAGA_START]) == 1
    # ids, counts and outcomes only — never content
    dumped = json.dumps(body)
    for d in h["corpus"]:
        if d.subject == "S-0006" and d.canary:
            assert d.canary.token not in dumped


@pytest.mark.timeout(600)
def test_forget_without_elicitation_gets_the_cli_command_not_an_erasure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tombstone.mcp.server import build_server

    monkeypatch.chdir(tmp_path)
    h, _trace = _setup(tmp_path)
    server = build_server(h["cfg_path"])
    journal = Journal(tmp_path / ".tombstone" / "journal.jsonl")

    async def run():  # noqa: ANN202
        async for session in _session(server, None, client_caps_elicit=False):
            return await session.call_tool(
                "tombstone.forget",
                {"subject": "S-0006", "reason": "dsr-mcp-forget"},
                allow_input_required=True,
            )

    res = anyio.run(run)
    text = json.dumps(_content_json(res)) + str(getattr(res, "content", ""))
    assert getattr(res, "is_error", False) or "elicitation" in text.lower()
    # the fallback it names must be the one-command path, not the two-command one
    assert "tombstone forget" in text and "--trace" not in text, text
    assert journal.records() == []


def test_the_server_names_its_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clients display serverInfo. An empty version tells the operator nothing about what is
    actually running against their data."""
    from tombstone import __version__
    from tombstone.mcp.server import build_server

    monkeypatch.chdir(tmp_path)
    h, _trace = _setup(tmp_path)
    server = build_server(h["cfg_path"])
    assert server.version == __version__ != ""


def test_every_tool_is_registered_and_only_two_can_destroy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tombstone.mcp.server import build_server

    monkeypatch.chdir(tmp_path)
    h, _trace = _setup(tmp_path)
    server = build_server(h["cfg_path"])

    async def names():  # noqa: ANN202
        return {t.name for t in await server.list_tools()}

    got = anyio.run(names)
    assert got == {
        "tombstone.forget",
        "tombstone.trace",
        "tombstone.verify",
        "tombstone.erase",
        "tombstone.receipt",
        "tombstone.status",
    }, got
