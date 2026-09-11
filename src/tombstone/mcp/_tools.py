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

from tombstone import __version__
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


def _confirmation_message(t: Any, reason: str) -> str:
    s = trace_summary(t)
    stores = ", ".join(f"{k}×{v}" for k, v in sorted(s["per_store"].items()))
    return (
        f"Erase {s['artifacts']} artifacts for subject {s['subject']} (reason {reason!r})? "
        f"Stores: {stores}. Lineage gaps: {len(s['gaps'])}. Third-party mentions: "
        f"{s['third_party_hits']} (never erased, listed for review). This is destructive and "
        "runs a suppress → reclaim → verify saga."
    )


def ask_confirmation(
    trace_id: str, reason: str, _ok: Annotated[bool, Resolve(client_can_confirm)]
) -> Elicit[ConfirmErase]:
    t = _rt().lineage.load_trace(trace_id)
    if t is None:
        raise ToolError(f"no trace {trace_id!r}; call tombstone.trace first")
    return Elicit(message=_confirmation_message(t, reason), schema=ConfirmErase)


# The trace each pending confirmation was shown for, so the erase runs on exactly what the
# operator approved rather than on whatever a second trace would find. Resolvers must be safe to
# re-run, and this one is: the trace id is derived from (subject, scope, snapshot), so re-tracing
# an unchanged graph yields the same id and overwrites the entry with itself.
_SHOWN: dict[str, str] = {}


def _shown_key(subject_hmac: str, reason: str) -> str:
    return hashlib.sha256(f"{subject_hmac}\x00{reason}".encode()).hexdigest()


def client_can_confirm_forget(ctx: Context, subject: str, reason: str) -> bool:
    caps = ctx.client_capabilities
    elicitation = caps.elicitation if caps is not None else None
    has_form = elicitation is not None and (elicitation.form is not None or elicitation.url is None)
    if not has_form:
        raise ToolError(
            "this client does not support form elicitation, so forget cannot be confirmed here "
            "and nothing was changed. Run: "
            f"tombstone forget {subject} --reason {reason}"
        )
    return True


def ask_forget_confirmation(
    subject: str, reason: str, _ok: Annotated[bool, Resolve(client_can_confirm_forget)]
) -> Elicit[ConfirmErase]:
    from tombstone.commands.trace import run_trace

    try:
        t, _ = run_trace(_rt(), subject)
    except TombstoneError as e:
        raise ToolError(str(e)) from e
    _SHOWN[_shown_key(t.subject.hmac, reason)] = t.trace_id
    return Elicit(message=_confirmation_message(t, reason), schema=ConfirmErase)


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


def tool_forget(
    subject: str,
    reason: str,
    decision: Annotated[ElicitationResult[ConfirmErase], Resolve(ask_forget_confirmation)],
) -> dict[str, Any]:
    """Trace a subject and erase everything descending from them, in one step. Asks for explicit
    confirmation first and erases exactly what that confirmation listed. On clients without
    elicitation, use the CLI: tombstone forget <subject> --reason <r>."""
    from tombstone.api import erase_traced
    from tombstone.commands.trace import run_trace
    from tombstone.model.artifacts import SubjectRef

    runtime = _rt()
    key = _shown_key(SubjectRef.from_raw(subject, runtime.pepper()).hmac, reason)
    shown = _SHOWN.pop(key, None)
    accepted = getattr(decision, "data", None)
    if (
        isinstance(decision, DeclinedElicitation)
        or accepted is None
        or not getattr(accepted, "confirm", False)
    ):
        t = runtime.lineage.load_trace(shown) if shown else None
        return {
            "resultType": "declined",
            "summary": trace_summary(t) if t is not None else {"subject": "", "artifacts": 0},
            "journal_written": False,
        }
    # Erase the trace the operator was shown, not a fresh one: a second trace could differ, and
    # then the receipt would not describe what was approved. If the graph moved in between, the
    # saga's own staleness check refuses rather than erasing the difference silently.
    t = runtime.lineage.load_trace(shown) if shown else None
    if t is None:  # a resolver that ran in another process (stateless HTTP): trace again
        try:
            t, _ = run_trace(runtime, subject)
        except TombstoneError as e:
            raise ToolError(str(e)) from e
    try:
        result = erase_traced(runtime, t, reason)
    except TombstoneError as e:
        raise ToolError(str(e)) from e
    return {
        "resultType": "receipt",
        "exit_code": result.exit_code,
        "receipt_id": result.receipt_id,
        "counts": dict(result.counts),
        "rendered": result.report,
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
        version=__version__,  # clients show this; an empty string tells the operator nothing
        instructions=(
            "Tombstone tracks where a data subject's data went (chunks, embeddings, caches, "
            "training examples, adapters), erases it everywhere, and returns a receipt that says "
            "what was checked and what was not. To act on a deletion request, call "
            "tombstone.forget with the subject id: it traces, asks the operator to confirm what "
            "it found, and erases exactly that. forget and erase are destructive and require "
            "explicit confirmation; trace, verify, receipt and status change nothing."
        ),
        request_state_security=RequestStateSecurity(keys=[state_key], ttl=600.0),
    )
    server.tool(name="tombstone.trace")(tool_trace)
    server.tool(name="tombstone.verify")(tool_verify)
    server.tool(name="tombstone.forget")(tool_forget)
    server.tool(name="tombstone.erase")(tool_erase)
    server.tool(name="tombstone.receipt")(tool_receipt)
    server.tool(name="tombstone.status")(tool_status)
    return server
