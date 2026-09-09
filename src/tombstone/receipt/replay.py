"""``tombstone replay``: re-derive every receipt's status table from the journal (Hard Rule 9).

Probe results are journaled, so the lattice can be re-run offline. A mismatch names the first
divergent receipt and artifact and is a build failure. A receipt written under a different
``semantics_version`` is reported (both versions) and exits non-zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tombstone.chain import Record
from tombstone.erase.journal import Journal
from tombstone.errors import ReplayMismatch
from tombstone.model.artifacts import ArtifactRef
from tombstone.model.status import SEMANTICS_VERSION, ArtifactStatus, Outcome, Receipt
from tombstone.receipt.ledger import Ledger
from tombstone.verify.levels import Facts, assign


@dataclass
class ReplayReport:
    receipts: int = 0
    matched: int = 0
    mismatches: list[str] = field(default_factory=list)
    version_mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.version_mismatches


def _content_matches(m: dict[str, Any]) -> float:
    return sum(
        float(v)
        for k, v in m.items()
        if k.startswith("matches_")
        and k not in {"matches_artifact_id", "matches_id", "matches_cache_key"}
    )


def facts_from_records(
    receipt: Receipt,
    status: ArtifactStatus,
    records: list[Record],
    saga_ids: set[str],
    this_saga: str,
) -> Facts:
    """Rebuild the Facts for one artifact purely from journal records (mirrors the saga)."""
    a: ArtifactRef = status.artifact
    suppressed_at: int | None = None
    for r in records:
        if (
            r.type == Journal.STEP_END
            and r.body.get("saga_id") == this_saga
            and r.body.get("step_id") == "suppress:lineage"
            and r.body.get("ok")
        ):
            suppressed_at = r.seq
            break
    dlq_error = None
    for r in records:
        if (
            r.type == Journal.DLQ
            and r.body.get("saga_id") == this_saga
            and a.artifact_id in r.body.get("artifact_ids", [])
        ):
            dlq_error = str(r.body.get("error", ""))
    reclaimed = any(
        r.type == Journal.STEP_END
        and r.body.get("saga_id") == this_saga
        and r.body.get("step_id") == f"reclaim:{a.store}"
        and r.body.get("ok")
        for r in records
    )
    f: dict[str, Any] = {
        "artifact": a,
        "trace_scope": receipt.scope,
        "suppressed_at": suppressed_at,
        "reclaimed": reclaimed,
        "dlq_error": dlq_error,
        "no_adapter": status.rule_id == "no_adapter",
        "lineage_gap": status.rule_id == "lineage_gap",
        "is_third_party": status.rule_id == "third_party",
        "physical_supported": True,
        "physical_reason": "",
        "model_applicable": False,
        "semantic_applicable": False,
    }
    extra: dict[str, float] = {}
    baseline_seen = False
    for r in records:
        if r.type != Journal.PROBE or r.body.get("artifact_id") != a.artifact_id:
            continue
        b = r.body
        if b.get("probe") == "baseline":
            if not baseline_seen and b.get("saga_id") in saga_ids:
                extra["baseline_content_matches"] = float(
                    b.get("measurement", {}).get("content_matches", 0.0)
                )
                extra["live_duplicates"] = float(
                    b.get("measurement", {}).get("live_duplicates", 0.0)
                )
                baseline_seen = True
            continue
        if b.get("saga_id") != this_saga:
            continue
        m = b.get("measurement", {})
        if b["level"] == "logical":
            f["logical_found"] = bool(b["found"])
        elif b["level"] == "physical":
            if b["probe"] == "unsupported":
                f["physical_supported"] = False
                f["physical_reason"] = b.get("detail", "")
            else:
                id_hits = float(m.get("matches_artifact_id", m.get("matches_id", 0.0)))
                content_hits = _content_matches(m)
                if id_hits > 0:
                    f["physical_found"] = True
                elif content_hits > 0 and extra.get("live_duplicates", 0.0) > 0:
                    baseline = extra.get("baseline_content_matches")
                    if baseline is not None and content_hits < baseline:
                        f["physical_found"] = False
                    else:
                        f["physical_found"] = None
                        f["physical_supported"] = False
                        f["physical_reason"] = "duplicate content"
                elif content_hits > 0:
                    f["physical_found"] = True
                else:
                    f["physical_found"] = False
        elif b["level"] == "model":
            f["model_applicable"] = True
            f["canary_rate"] = m.get("canary_rate")
            f["canary_extracted"] = int(m["canary_extracted"]) if "canary_extracted" in m else None
            f["canary_total"] = int(m["canary_total"]) if "canary_total" in m else None
            f["mia_auc"] = m.get("mia_auc")
            f["mia_ci_low"] = m.get("mia_ci_low")
            f["mia_ci_high"] = m.get("mia_ci_high")
        elif b["level"] == "semantic":
            f["semantic_applicable"] = True
            f["drift"] = m.get("drift")
            f["control"] = m.get("control")
            f["drift_ci_low"] = m.get("drift_ci_low")
            f["drift_ci_high"] = m.get("drift_ci_high")
            f["control_ci_low"] = m.get("control_ci_low")
            f["control_ci_high"] = m.get("control_ci_high")
            f["query_budget"] = int(m["query_budget"]) if "query_budget" in m else None
    if status.rule_id in {"no_adapter", "dlq", "lineage_gap", "third_party"}:
        # these facts are not probe-derived; the saga knew them structurally
        pass
    if f["no_adapter"] or f["lineage_gap"]:
        f["physical_supported"] = status.rule_id != "physical_unsupported"
    return Facts(**f)


def replay_receipt(receipt: Receipt, journal: Journal) -> list[str]:
    """Return a list of divergences (empty = match) for one receipt."""
    records = journal.records()
    starts = {r.body["saga_id"]: r.body for r in records if r.type == Journal.SAGA_START}
    ends = {
        r.body.get("receipt_id"): r.body["saga_id"] for r in records if r.type == Journal.SAGA_END
    }
    saga_id = ends.get(receipt.receipt_id)
    if saga_id is None:
        return [f"receipt {receipt.receipt_id}: no saga_end record in the journal"]
    saga_ids = {s for s, b in starts.items() if b.get("trace_id") == receipt.trace_id}
    problems: list[str] = []
    for status in receipt.statuses:
        facts = facts_from_records(receipt, status, records, saga_ids, saga_id)
        try:
            derived = assign(facts)
        except Exception as e:
            problems.append(
                f"receipt {receipt.receipt_id} artifact {status.artifact.artifact_id}: replay raised {type(e).__name__}: {e}"
            )
            continue
        if (derived.outcome, derived.level, derived.rule_id) != (
            status.outcome,
            status.level,
            status.rule_id,
        ):
            problems.append(
                f"receipt {receipt.receipt_id} artifact {status.artifact.artifact_id}: recorded "
                f"{status.outcome.value}/{status.level.value if status.level else '-'}/{status.rule_id} "
                f"but replay derives {derived.outcome.value}/{derived.level.value if derived.level else '-'}/{derived.rule_id}"
            )
    # counts must agree with statuses
    recount: dict[Outcome, int] = dict.fromkeys(Outcome, 0)
    for s in receipt.statuses:
        recount[s.outcome] += 1
    recount[Outcome.NEEDS_HUMAN] += len(receipt.needs_human)
    for o in Outcome:
        if recount[o] != receipt.counts.get(o, 0):
            problems.append(
                f"receipt {receipt.receipt_id}: count {o.value} recorded {receipt.counts.get(o, 0)} but statuses give {recount[o]}"
            )
    return problems


def replay_ledger(
    ledger_path: Path, journal_path: Path, lattice_version: int = SEMANTICS_VERSION
) -> ReplayReport:
    ledger = Ledger(ledger_path)
    journal = Journal(journal_path)
    ledger.verify()
    journal.verify()
    report = ReplayReport()
    for receipt in ledger.receipts():
        report.receipts += 1
        if receipt.semantics_version != lattice_version:
            report.version_mismatches.append(
                f"receipt {receipt.receipt_id}: written under semantics v{receipt.semantics_version}, "
                f"this build replays v{lattice_version}"
            )
            continue
        problems = replay_receipt(receipt, journal)
        if problems:
            report.mismatches.extend(problems)
        else:
            report.matched += 1
    return report


def assert_replay(ledger_path: Path, journal_path: Path) -> ReplayReport:
    report = replay_ledger(ledger_path, journal_path)
    if not report.ok:
        first = (report.mismatches or report.version_mismatches)[0]
        raise ReplayMismatch(
            f"replay diverged ({len(report.mismatches)} mismatch(es), {len(report.version_mismatches)} version mismatch(es)); first: {first}"
        )
    return report
