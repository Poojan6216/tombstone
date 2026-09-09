"""Audit mode: ``tombstone verify --subject`` — probe every artifact of a subject at every layer
the store permits, *without* erasing anything. This is Demo 1: what native ``delete()`` reached.

Per (kind, store) row:
  GONE     logical pass and physical pass (or physical not applicable)
  HIDDEN   logical pass, physical FAIL — the bytes are still there
  PRESENT  logical FAIL — still retrievable
  UNVERIFIED(reason) when a layer cannot be checked
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tombstone.errors import NotSupported
from tombstone.model.artifacts import ArtifactKind, ArtifactRef
from tombstone.model.lineage import Trace
from tombstone.model.status import VerifyLevel
from tombstone.receipt.render import render_audit
from tombstone.registry import Runtime
from tombstone.stores.base import ProbeSet
from tombstone.verify.logical import build_probe_set


@dataclass
class ArtifactAudit:
    artifact: ArtifactRef
    logical_found: bool | None
    logical_by: tuple[str, ...]
    physical_found: bool | None
    physical_reason: str
    physical_detail: str
    model: dict[str, Any] = field(default_factory=dict)
    extra: list[str] = field(default_factory=list)

    @property
    def state(self) -> str:
        if self.logical_found is None:
            return "UNVERIFIED"
        if self.logical_found:
            return "PRESENT"
        if self.physical_found is None:
            return "GONE" if self.physical_reason == "n/a" else "HIDDEN?"
        return "HIDDEN" if self.physical_found else "GONE"

    @property
    def recoverable(self) -> bool:
        return self.state in {"PRESENT", "HIDDEN"} or bool(self.model.get("found"))


@dataclass
class AuditReport:
    trace: Trace
    audits: list[ArtifactAudit]

    @property
    def recoverable(self) -> int:
        return sum(1 for a in self.audits if a.recoverable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace.trace_id,
            "subject": self.trace.subject.hmac,
            "artifacts": len(self.audits),
            "recoverable": self.recoverable,
            "rows": [
                {
                    "artifact_id": a.artifact.artifact_id,
                    "kind": a.artifact.kind.value,
                    "store": a.artifact.store,
                    "state": a.state,
                    "logical_found": a.logical_found,
                    "logical_by": list(a.logical_by),
                    "physical_found": a.physical_found,
                    "physical_reason": a.physical_reason,
                    "physical_detail": a.physical_detail,
                    "model": a.model,
                }
                for a in self.audits
            ],
        }


def audit_trace(
    rt: Runtime, trace: Trace, probe_prompts: list[tuple[str, str]] | None = None
) -> AuditReport:
    audits: list[ArtifactAudit] = []
    for a in trace.artifacts:
        if a.store not in {s.name for s in rt.cfg.stores}:
            audits.append(ArtifactAudit(a, None, (), None, "no adapter configured", ""))
            continue
        store = rt.store(a.store)
        setter = getattr(store, "set_context", None)
        if callable(setter):
            setter(trace)
        if probe_prompts is not None and hasattr(store, "probe_prompts"):
            store.probe_prompts = probe_prompts
        dims = int(getattr(store, "dims", 0) or 0)
        probes = build_probe_set(rt.lineage, a, dims, rt.cfg.erase.logical_probe_k)
        lr = store.probe_logical(
            a,
            probes
            if probes.vectors or a.kind is not ArtifactKind.EMBED
            else ProbeSet(a.artifact_id),
        )
        physical_found: bool | None = None
        physical_reason = "n/a"
        physical_detail = ""
        if VerifyLevel.PHYSICAL in store.capabilities:
            try:
                pr = store.probe_physical(a)
                physical_found = pr.found
                physical_reason = ""
                physical_detail = ";".join(pr.locations) if pr.found else pr.detail
            except NotSupported as e:
                physical_reason = str(e)
        else:
            reason = getattr(store, "physical_unsupported_reason", None)
            physical_reason = reason() if callable(reason) else "store lacks PHYSICAL capability"
        model: dict[str, Any] = {}
        if a.kind is ArtifactKind.ADAPTER and hasattr(store, "probe_model"):
            model = store.probe_model(a)
        audits.append(
            ArtifactAudit(
                a, lr.found, lr.found_by, physical_found, physical_reason, physical_detail, model
            )
        )
    return AuditReport(trace, audits)


def render_audit_report(report: AuditReport) -> str:
    groups: dict[tuple[str, str], list[ArtifactAudit]] = {}
    for a in report.audits:
        groups.setdefault((a.artifact.kind.value, a.artifact.store), []).append(a)
    order = ["source", "chunk", "embed", "cache", "train", "adapter", "memory"]
    rows: list[tuple[str, str, str, str, str]] = []
    for kind, store in sorted(
        groups, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1])
    ):
        items = groups[(kind, store)]
        n = len(items)
        states = {a.state for a in items}
        state = (
            "PRESENT"
            if "PRESENT" in states
            else (
                "HIDDEN"
                if "HIDDEN" in states
                else (
                    "UNVERIFIED"
                    if "UNVERIFIED" in states
                    else ("HIDDEN?" if "HIDDEN?" in states else "GONE")
                )
            )
        )
        logical_pass = sum(1 for a in items if a.logical_found is False)
        physical_fail = sum(1 for a in items if a.physical_found)
        detail = ""
        extra_lines: list[str] = []
        if state == "PRESENT":
            by = sorted({b.split(":")[0] for a in items for b in a.logical_by})
            detail = f"still retrievable by {', '.join(by)}" if by else "still retrievable"
            if kind == "cache":
                detail = "cached answer still returned" + (
                    " on paraphrased query"
                    if any("topk" in b for a in items for b in a.logical_by)
                    else ""
                )
        elif state == "HIDDEN":
            detail = "logical PASS, physical FAIL"
            locs = sorted(
                {
                    loc.split(":")[0]
                    for a in items
                    for loc in (a.physical_detail.split(";") if a.physical_detail else [])
                }
            )
            if locs:
                extra_lines.append(
                    f"bytes for {physical_fail}/{n} ids located in {' + '.join(locs)}"
                )
            ms = [a for a in items if "dead_tuples" in str(a.physical_detail)]
            _ = ms
        elif state == "GONE":
            lvl = "physical" if any(a.physical_found is False for a in items) else "logical"
            detail = f"({lvl})"
        elif state == "HIDDEN?":
            detail = f"logical PASS, physical UNVERIFIED ({items[0].physical_reason[:60]})"
        else:
            detail = f"({items[0].physical_reason or 'no adapter'})"
        if kind == "adapter" and items and items[0].model:
            m = items[0].model
            meas = m.get("measurement", {})
            ce, ct = int(meas.get("canary_extracted", 0)), int(meas.get("canary_total", 0))
            rate = f"{round(100 * meas.get('canary_rate', 0.0))}%"
            state = "PRESENT" if m.get("found") else "GONE"
            detail = (
                f"canary extracted at {rate} ({ce}/{ct} prefixes, greedy)"
                if ct
                else "no extraction prompts"
            )
            if "mia_auc" in meas:
                extra_lines.append(
                    f"MIA AUC {meas['mia_auc']:.2f} [CI {meas['mia_ci_low']:.2f},{meas['mia_ci_high']:.2f}]"
                )
        _ = logical_pass
        rows.append((f"{kind:<7} ×{n}", store, state, detail, "\n".join(extra_lines)))
    total = len(report.audits)
    rec = report.recoverable
    verdict = (
        f"NOT ERASED.  {rec}/{total} artifacts still hold recoverable content."
        if rec
        else f"no recoverable content found at any checked layer ({total} artifacts)."
    )
    return render_audit(rows, report.trace.subject.short, total, verdict)
