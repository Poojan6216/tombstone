"""The public Python API: two calls, no command line.

A deletion request does not arrive in a terminal. It arrives in the operator's own product — a
customer clicks "delete my account", a ticket lands in a support queue — so the erasure belongs
in the code that already handles that:

    from tombstone import forget

    result = forget("S-0417", reason=f"dsr-{ticket_id}")
    if not result.ok:
        alert_privacy_team(result.report)   # something could not be confirmed; it says what

and, to look without touching anything:

    held = trace("S-0417")
    print(held.summary())

These are thin wrappers over the same code paths ``tombstone trace`` and ``tombstone erase``
take. The lineage graph, the journal, the receipt and the exit codes are identical whichever
entry point you come in through, so a receipt produced from Python is indistinguishable from one
produced at the terminal, and ``tombstone replay`` re-derives both.

Both functions use ``Runtime.shared``: one runtime per config path per process, so calling them
from an app that already has a ``TombstoneVectorStore`` open reuses that store's adapters rather
than opening a second client against the same files.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tombstone.model.lineage import Trace
    from tombstone.registry import Runtime

__all__ = ["Erasure", "Held", "forget", "trace"]


@dataclass(frozen=True)
class Held:
    """What a subject's data has become. The answer to "what do we hold on this person?".

    Nothing here is content: stores, kinds and counts only, and the subject as its HMAC prefix —
    the raw id is never stored or logged (Hard Rule 3).
    """

    subject: str
    trace_id: str
    scope: str
    count: int
    by_store: Mapping[str, int]
    by_kind: Mapping[str, int]
    gaps: tuple[str, ...] = ()
    needs_human: int = 0
    shared: int = 0
    already_tombstoned: int = 0

    @property
    def empty(self) -> bool:
        return self.count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "trace_id": self.trace_id,
            "scope": self.scope,
            "count": self.count,
            "by_store": dict(self.by_store),
            "by_kind": dict(self.by_kind),
            "gaps": list(self.gaps),
            "needs_human": self.needs_human,
            "shared": self.shared,
            "already_tombstoned": self.already_tombstoned,
        }

    def summary(self) -> str:
        """The preview a human is asked to approve. Plain text, no colour, safe to log."""
        if self.empty:
            head = f"Nothing is held for subject {self.subject} in scope {self.scope}."
        else:
            noun = "thing" if self.count == 1 else "things"
            head = (
                f"{self.count} {noun} exist because of subject {self.subject}"
                f"   (raw id never stored)"
            )
        lines = [head, ""]
        for store in sorted(self.by_store):
            lines.append(f"  {store:<30} {self.by_store[store]:>4}")
        if self.by_kind:
            lines.append("")
            lines.append(
                "  kinds: " + ", ".join(f"{k}×{v}" for k, v in sorted(self.by_kind.items()))
            )
        if self.already_tombstoned:
            lines.append(f"  already erased in an earlier request: {self.already_tombstoned}")
        if self.needs_human:
            lines.append("")
            lines.append(
                f"  ! {self.needs_human} document(s) belong to other people and mention this "
                "subject.\n    They are listed for human review and are never erased "
                "automatically: deleting\n    another person's record would be a different "
                "violation."
            )
        lines.append("")
        if self.gaps:
            lines.append(f"  ! lineage gaps: {len(self.gaps)} — erase will refuse without")
            lines.append("    accept_gaps=True, because data that arrived before Tombstone was")
            lines.append("    installed has no trail and cannot be claimed as erased:")
            for g in self.gaps:
                lines.append(f"      {g}")
        else:
            lines.append("  lineage gaps: none")
        return "\n".join(lines)


@dataclass(frozen=True)
class Erasure:
    """The outcome of a ``forget``: what was checked, what was not, and where the receipt is.

    ``ok`` is true only when every artifact was VERIFIED. It is false for the honest middle
    cases — a managed database the tool cannot read the files of, bytes shared with another live
    record — which are outcomes, not errors, and each carries its reason in ``report``.
    """

    ok: bool
    exit_code: int
    subject: str
    trace_id: str
    receipt_id: str
    receipt_path: Path
    counts: Mapping[str, int] = field(default_factory=dict)
    report: str = ""

    @property
    def verified(self) -> int:
        return int(self.counts.get("verified", 0))

    @property
    def unverified(self) -> int:
        return int(self.counts.get("unverified", 0))

    @property
    def residual(self) -> int:
        return int(self.counts.get("residual", 0))

    @property
    def out_of_scope(self) -> int:
        return int(self.counts.get("out_of_scope", 0))

    @property
    def needs_human(self) -> int:
        return int(self.counts.get("needs_human", 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "subject": self.subject,
            "trace_id": self.trace_id,
            "receipt_id": self.receipt_id,
            "receipt_path": str(self.receipt_path),
            "counts": dict(self.counts),
        }

    def summary(self) -> str:
        return self.report


# --- internals shared with the CLI and the MCP server --------------------------------------------


def held_from_trace(t: Trace) -> Held:
    by_store: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for a in t.artifacts:
        by_store[a.store] = by_store.get(a.store, 0) + 1
        by_kind[a.kind.value] = by_kind.get(a.kind.value, 0) + 1
    return Held(
        subject=t.subject.short,
        trace_id=t.trace_id,
        scope=t.scope.tenant,
        count=len(t.artifacts),
        by_store=by_store,
        by_kind=by_kind,
        gaps=tuple(t.gaps),
        needs_human=len(t.third_party_hits),
        shared=len(t.shared),
        already_tombstoned=len(t.already_tombstoned),
    )


def look(rt: Runtime, subject: str, scan_stores: bool = True) -> tuple[Held, Trace]:
    """Trace a subject on an already-open runtime. Returns the summary and the trace itself."""
    from tombstone.commands.trace import run_trace

    t, _report = run_trace(rt, subject, with_store_gaps=scan_stores)
    return held_from_trace(t), t


def erase_traced(
    rt: Runtime,
    t: Trace,
    reason: str,
    accept_gaps: bool = False,
    reclaim: bool = True,
    semantic: bool = False,
) -> Erasure:
    """Erase an already-traced subject on an open runtime. The confirmation is the caller's:
    the CLI prompts, the MCP server elicits, and a direct call from application code is itself
    the deliberate act."""
    from tombstone.commands.erase import run_erase

    code, text, data = run_erase(
        rt,
        t.trace_id,
        reason,
        confirm=True,
        accept_gaps=accept_gaps,
        reclaim=reclaim,
        semantic=semantic,
    )
    receipt_id = str(data["receipt_id"])
    raw_counts = data.get("counts")
    counts = {str(k): int(v) for k, v in raw_counts.items()} if isinstance(raw_counts, dict) else {}
    return Erasure(
        ok=code == 0,
        exit_code=code,
        subject=t.subject.short,
        trace_id=t.trace_id,
        receipt_id=receipt_id,
        receipt_path=rt.inst.receipts_dir / f"{receipt_id}.json",
        counts=counts,
        report=text,
    )


# --- the public two ------------------------------------------------------------------------------


def trace(subject: str, *, config: str | Path | None = None, scan_stores: bool = True) -> Held:
    """What does this installation hold because of ``subject``? Reads only; changes nothing.

    ``subject`` is the raw id your application knows (a customer id, an email). It is hashed
    under this installation's pepper before it touches the database and is never stored raw.
    ``scan_stores`` samples each store for entries with no lineage and reports them as gaps; pass
    False to skip that (faster, and it will not notice data that arrived unwrapped).

    Raises :class:`~tombstone.errors.LineageGapError` for a subject with no lineage records at
    all, rather than returning an empty result. "We hold nothing on this person" and "this person
    arrived before we were watching" look identical from inside the database, and reporting the
    reassuring one as fact is the failure mode this tool exists to prevent (Hard Rule 4). Catch
    it if a not-found answer is what you want; the message says which question is unanswered.
    """
    from tombstone.registry import Runtime

    rt = Runtime.shared(config)
    held, _t = look(rt, subject, scan_stores=scan_stores)
    return held


def forget(
    subject: str,
    *,
    reason: str,
    config: str | Path | None = None,
    accept_gaps: bool = False,
    reclaim: bool = True,
    semantic: bool = False,
) -> Erasure:
    """Trace ``subject`` and erase everything that descends from them. **This is destructive.**

    Calling it is the confirmation — there is no flag to pass, because a call written in your own
    source code is already a deliberate act, and a boolean people paste without reading protects
    nobody. ``reason`` is required and goes on the receipt; use the DSR ticket id.

    Returns an :class:`Erasure` describing what was checked. It does not raise when something
    could not be confirmed — a managed database, bytes shared with a live record — because those
    are outcomes with reasons, not failures; check ``result.ok`` and read ``result.report``.

    Raises :class:`~tombstone.errors.LineageGapError` in two cases, both deliberate. A subject
    with no lineage records at all raises whatever you pass, because there is nothing to erase
    *and no way to know that* (see :func:`trace`). A subject whose stores hold entries with no
    lineage raises unless ``accept_gaps=True``, which proceeds and records UNVERIFIED(lineage-gap)
    on the receipt rather than quietly claiming an erasure the graph cannot support.
    """
    from tombstone.registry import Runtime

    rt = Runtime.shared(config)
    _held, t = look(rt, subject, scan_stores=True)
    return erase_traced(rt, t, reason, accept_gaps=accept_gaps, reclaim=reclaim, semantic=semantic)
