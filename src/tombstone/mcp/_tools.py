"""Tool definitions for the MCP server. Imported only when the ``mcp`` extra is installed.

The SDK evaluates tool annotations from this module's globals (``Annotated[..., Resolve(...)]``),
so every name used in a signature is imported here at module level.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import (
    Context,
    DeclinedElicitation,
    Elicit,
    ElicitationResult,
    MCPServer,
    RequestStateSecurity,
    Resolve,
)
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from tombstone.errors import TombstoneError
from tombstone.registry import Runtime


class ConfirmErase(BaseModel):
    confirm: bool = Field(description="Set true to erase; anything else cancels.")


def trace_summary(t: Any) -> dict[str, Any]:
    by_store: dict[str, int] = {}
    per_kind: dict[str, int] = {}
    for a in t.artifacts:
        by_store[a.store] = by_store.get(a.store, 0) + 1
        per_kind[a.kind.value] = per_kind.get(a.kind.value, 0) + 1
    return {
        "trace_id": t.trace_id,
        "subject": t.subject.short,
        "scope": t.scope.tenant,
        "artifacts": len(t.artifacts),
        "per_store": by_store,
        "per_kind": per_kind,
        "gaps": list(t.gaps),
        "third_party_hits": len(t.third_party_hits),
        "shared": len(t.shared),
    }


_CONFIG: dict[str, str | Path | None] = {"path": None}


def _rt() -> Runtime:
    return Runtime.shared(_CONFIG["path"])


# --- resolvers for erase ------------------------------------------------------------------------


def client_can_confirm(ctx: Context, trace_id: str, reason: str) -> bool:
    caps = ctx.client_capabilities
    elicitation = caps.elicitation if caps is not None else None
    has_form = elicitation is not None and (elicitation.form is not None or elicitation.url is None)
    if not has_form:
        raise ToolError(
            "this client does not support form elicitation, so erase cannot be confirmed here and "
            "nothing was changed. Run: "
            f"tombstone erase --trace {trace_id} --reason {reason} --confirm"
        )
    return True


def ask_confirmation(
    trace_id: str, reason: str, _ok: Annotated[bool, Resolve(client_can_confirm)]
) -> Elicit[ConfirmErase]:
    t = _rt().lineage.load_trace(trace_id)
    if t is None:
        raise ToolError(f"no trace {trace_id!r}; call tombstone.trace first")
    s = trace_summary(t)
    stores = ", ".join(f"{k}×{v}" for k, v in sorted(s["per_store"].items()))
    message = (
        f"Erase {s['artifacts']} artifacts for subject {s['subject']} (reason {reason!r})? "
        f"Stores: {stores}. Lineage gaps: {len(s['gaps'])}. Third-party mentions: "
        f"{s['third_party_hits']} (never erased, listed for review). This is destructive and "
        "runs a suppress → reclaim → verify saga."
    )
    return Elicit(message=message, schema=ConfirmErase)


# --- tools ----------------------------------------------------------------------------------


def tool_trace(subject: str, scope: str | None = None) -> dict[str, Any]:
    """List every artifact descending from a subject (ids and stores, never content)."""
    from tombstone.commands.trace import run_trace

    runtime = _rt()
    if scope and scope != runtime.scope.tenant:
        raise ToolError(f"this installation serves scope {runtime.scope.tenant!r}; got {scope!r}")
    try:
        t, _ = run_trace(runtime, subject)
    except TombstoneError as e:
        raise ToolError(str(e)) from e
    return trace_summary(t)


def tool_verify(trace_id: str) -> dict[str, Any]:
    """Probe every artifact of a trace at every layer the store permits, without erasing."""
    from tombstone.verify.audit import audit_trace

    runtime = _rt()
    t = runtime.lineage.load_trace(trace_id)
    if t is None:
        raise ToolError(f"no trace {trace_id!r}")
    return audit_trace(runtime, t).to_dict()


def tool_erase(
    trace_id: str,
    reason: str,
    decision: Annotated[ElicitationResult[ConfirmErase], Resolve(ask_confirmation)],
) -> dict[str, Any]:
    """Two-phase erasure of a traced subject. Asks for explicit confirmation first; on clients
    without elicitation, use the CLI: tombstone erase --trace <id> --reason <r> --confirm."""
    from tombstone.commands.erase import run_erase

    runtime = _rt()
    t = runtime.lineage.load_trace(trace_id)
    if t is None:
        raise ToolError(f"no trace {trace_id!r}; call tombstone.trace first")
    accepted = getattr(decision, "data", None)
    if (
        isinstance(decision, DeclinedElicitation)
        or accepted is None
        or not getattr(accepted, "confirm", False)
    ):
        return {"resultType": "declined", "summary": trace_summary(t), "journal_written": False}
    try:
        code, text, data = run_erase(runtime, trace_id, reason, confirm=True, accept_gaps=False)
    except TombstoneError as e:
        raise ToolError(str(e)) from e
    return {
        "resultType": "receipt",
        "exit_code": code,
        "receipt_id": data["receipt_id"],
        "counts": data["counts"],
        "rendered": text,
    }


def tool_receipt(receipt_id: str) -> dict[str, Any]:
    """Return a receipt (statuses carry ids, stores, outcomes and measurements; never content)."""
    from tombstone.receipt.ledger import Ledger

    r = Ledger(_rt().inst.ledger_path).find(receipt_id)
    if r is None:
        raise ToolError(f"no receipt {receipt_id!r}")
    return r.to_dict()


def tool_status() -> dict[str, Any]:
    """Lineage coverage per store, pins, journal and DLQ depth."""
    from tombstone.commands.trace import status_payload

    return status_payload(_rt())


def build(config: str | Path | None) -> MCPServer:
    _CONFIG["path"] = config
    runtime = Runtime.shared(config)
    state_key = hashlib.sha256(b"tombstone-mcp-request-state" + runtime.pepper()).digest()
    server: MCPServer = MCPServer(
        "tombstone",
        instructions=(
            "Tombstone tracks where a data subject's data went (chunks, embeddings, caches, "
            "training examples, adapters), erases it everywhere, and returns a receipt that says "
            "what was checked and what was not. erase is destructive and requires explicit "
            "confirmation."
        ),
        request_state_security=RequestStateSecurity(keys=[state_key], ttl=600.0),
    )
    server.tool(name="tombstone.trace")(tool_trace)
    server.tool(name="tombstone.verify")(tool_verify)
    server.tool(name="tombstone.erase")(tool_erase)
    server.tool(name="tombstone.receipt")(tool_receipt)
    server.tool(name="tombstone.status")(tool_status)
    return server
