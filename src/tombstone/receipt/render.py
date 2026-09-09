"""Render the §2 receipt block. The renderer performs no arithmetic: every number it prints was
computed by the saga and stored in the receipt or the view passed in."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from tombstone.model.status import Outcome, Receipt, VerifyLevel


@dataclass(frozen=True, slots=True)
class StoreRow:
    store: str
    method: str  # reclaim method string from the store
    level: str  # "physical" | "logical" | "model" | "-"
    outcome: str  # "VERIFIED" | "UNVERIFIED-managed" | "RESIDUAL" | ...
    detail: str  # e.g. "(bytes absent, 3/3)"
    note: str = ""  # a second line, e.g. the actionable hint


@dataclass(frozen=True, slots=True)
class SemanticRow:
    store: str
    drift: float
    control: float
    verdict: str  # "at control level" | "above control"


@dataclass(frozen=True, slots=True)
class ReceiptView:
    receipt_path: str
    suppressed: int
    total: int
    rows: tuple[StoreRow, ...]
    semantic: tuple[SemanticRow, ...] = ()
    needs_human: tuple[str, ...] = ()
    chain_ok: bool = True
    signed: bool = True
    exit_code: int = 0
    dlq: tuple[str, ...] = field(default=())


OUTCOME_LABEL = {
    Outcome.VERIFIED: "VERIFIED",
    Outcome.UNVERIFIED: "UNVERIFIED",
    Outcome.RESIDUAL: "RESIDUAL",
    Outcome.OUT_OF_SCOPE: "OUT_OF_SCOPE",
    Outcome.NEEDS_HUMAN: "NEEDS_HUMAN",
}


def render_receipt(receipt: Receipt, view: ReceiptView) -> str:
    lines: list[str] = []
    lines.append(
        f"phase 1 suppress    {view.suppressed}/{view.total} artifacts tombstoned"
        f"         (retrieval filter active)"
    )
    lines.append("phase 2 reclaim")
    for r in view.rows:
        head = f"  {r.store:<17} {r.method:<34} {r.level + ' ' + r.outcome:<26} {r.detail}"
        lines.append(head.rstrip())
        if r.note:
            lines.append(f"  {'':<17} {r.note}")
    lines.append("phase 3 verify")
    if view.semantic:
        first = True
        for s in view.semantic:
            label = "  semantic residue" if first else "                  "
            first = False
            lines.append(
                f"{label} {s.store:<14} drift {s.drift:.3f} vs same-cluster control {s.control:.3f}"
                f"   → {s.verdict}"
            )
    else:
        lines.append("  semantic residue  not measured")
    if view.needs_human:
        lines.append(
            f"  third-party mentions: {len(view.needs_human)} documents owned by other subjects "
            "contain this subject's name"
        )
        lines.append(
            "                    NOT erased (no lineage edge; would require editing other "
            "subjects' data). Listed for human review."
        )
    if view.dlq:
        lines.append(f"  dead-letter queue: {len(view.dlq)} step(s) failed — `tombstone dlq retry`")
    lines.append("")
    sig = "ed25519 signed" if view.signed else "UNSIGNED"
    chain = "chain ok" if view.chain_ok else "CHAIN BROKEN"
    rid = receipt.receipt_id
    lines.append(f"receipt: {view.receipt_path}   {sig}   {chain}")
    c = receipt.counts
    oos = ", ".join(_short(s) for s in receipt.out_of_scope)
    summary = (
        f"  VERIFIED {c.get(Outcome.VERIFIED, 0):>3}   UNVERIFIED {c.get(Outcome.UNVERIFIED, 0)}"
        f"   RESIDUAL {c.get(Outcome.RESIDUAL, 0)}   OUT_OF_SCOPE {len(receipt.out_of_scope)} ({oos})"
    )
    if c.get(Outcome.NEEDS_HUMAN, 0):
        summary += f"   NEEDS_HUMAN {c.get(Outcome.NEEDS_HUMAN, 0)}"
    lines.append(summary)
    lines.append(f"  id {rid}   semantics v{receipt.semantics_version}")
    lines.append(
        "  A receipt is a record of what was done and checked. It is not a legal instrument."
    )
    if view.exit_code:
        lines.append(f"exit {view.exit_code}")
    return "\n".join(lines)


def _short(layer: str) -> str:
    table = {
        "database backups and snapshots": "backups",
        "write-ahead logs and replicas": "WAL",
        "embedding-provider-side request logs": "provider-side embeddings",
    }
    return table.get(layer, layer)


def render_audit(
    rows: Sequence[tuple[str, str, str, str, str]], subject_short: str, total: int, verdict: str
) -> str:
    """Demo 1 table (``tombstone verify --subject``). Rows are precomputed strings."""
    lines = [
        f"subject: {subject_short}   (raw id never logged)",
        f"artifacts descending from this subject: {total}",
        "",
    ]
    for kind, store, state, detail, extra in rows:
        lines.append(f"  {kind:<8} {store:<22} {state:<10} {detail}".rstrip())
        if extra:
            for e in extra.split("\n"):
                lines.append(f"  {'':<8} {'':<22} {'':<10}   {e}")
    lines.append("")
    lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


LEVEL_LABEL = {
    VerifyLevel.LOGICAL: "logical",
    VerifyLevel.PHYSICAL: "physical",
    VerifyLevel.SEMANTIC: "semantic",
    VerifyLevel.MODEL: "model",
}
