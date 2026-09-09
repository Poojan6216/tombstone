"""The status lattice (§7). Pure: facts in, one ``ArtifactStatus`` out, first matching rule wins.

| order | rule_id              | condition                                     | outcome                 |
|-------|----------------------|-----------------------------------------------|-------------------------|
| 1     | scope_violation      | artifact scope ≠ trace scope                  | raise (Hard Rule 8)     |
| 2     | not_suppressed       | no suppression journal entry                  | raise (Hard Rule 5)     |
| 2b    | dlq                  | the store's step is dead-lettered             | UNVERIFIED(dlq)         |
| 2c    | no_adapter           | no adapter for the artifact's store           | UNVERIFIED(no adapter)  |
| 2d    | lineage_gap          | erased under --accept-gaps, store named in gap| UNVERIFIED(lineage-gap) |
| 3     | third_party          | SOURCE owned by another subject               | NEEDS_HUMAN             |
| 4     | logical_fail         | any logical probe returns the artifact        | RESIDUAL(logical)       |
| 5     | physical_unsupported | store lacks PHYSICAL capability               | UNVERIFIED(physical)    |
| 6     | physical_fail        | byte pattern / dead tuple found               | RESIDUAL(physical)      |
| 7     | model_residual       | adapter: canary rate > 0 or MIA CI excludes .5| RESIDUAL(model)         |
| 8     | semantic_residual    | drift CI excludes control                     | RESIDUAL(semantic)      |
| 9     | verified             | all applicable probes pass                    | VERIFIED(highest level) |

Rule 5 before rule 6 is the honesty of the tool: a managed store shows up as UNVERIFIED, never
as VERIFIED by omission. Rule 8 is reported, never blocks (a drift number is a measurement of the
neighbourhood, not proof the content is present) — it still yields RESIDUAL so the reader sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tombstone.errors import SagaError, ScopeViolation
from tombstone.model.artifacts import ArtifactRef, Scope
from tombstone.model.status import ArtifactStatus, Outcome, VerifyLevel

LATTICE_VERSION = 1


@dataclass(frozen=True, slots=True)
class Facts:
    """Everything the lattice may look at for one artifact. Produced by the saga from journaled
    probe records, so replay can rebuild it byte-for-byte."""

    artifact: ArtifactRef
    trace_scope: Scope
    suppressed_at: int | None  # journal seq of the durable suppression record
    reclaimed: bool = False
    reclaim_method: str = ""
    is_third_party: bool = False
    dlq_error: str | None = None
    no_adapter: bool = False
    lineage_gap: bool = False
    logical_found: bool | None = None  # None: not probed
    logical_detail: str = ""
    physical_supported: bool = False
    physical_reason: str = ""
    physical_found: bool | None = None
    physical_detail: str = ""
    model_applicable: bool = False
    model_probes_run: int = 0  # extraction prompts actually decoded
    canary_rate: float | None = None
    canary_extracted: int | None = None
    canary_total: int | None = None
    mia_auc: float | None = None
    mia_ci_low: float | None = None
    mia_ci_high: float | None = None
    semantic_applicable: bool = False
    drift: float | None = None
    control: float | None = None
    drift_ci_low: float | None = None
    drift_ci_high: float | None = None
    control_ci_low: float | None = None
    control_ci_high: float | None = None
    query_budget: int | None = None
    measurement_extra: dict[str, float] = field(default_factory=dict)


def _m(facts: Facts) -> dict[str, float]:
    m: dict[str, float] = dict(facts.measurement_extra)
    for key in (
        "canary_rate",
        "mia_auc",
        "mia_ci_low",
        "mia_ci_high",
        "drift",
        "control",
        "drift_ci_low",
        "drift_ci_high",
        "control_ci_low",
        "control_ci_high",
    ):
        v = getattr(facts, key)
        if v is not None:
            m[key] = float(v)
    if facts.canary_extracted is not None:
        m["canary_extracted"] = float(facts.canary_extracted)
    if facts.canary_total is not None:
        m["canary_total"] = float(facts.canary_total)
    if facts.query_budget is not None:
        m["query_budget"] = float(facts.query_budget)
    return m


def assign(facts: Facts) -> ArtifactStatus:
    a = facts.artifact
    # 1. scope
    if a.scope != facts.trace_scope:
        raise ScopeViolation(
            f"artifact {a.artifact_id} is in scope {a.scope.tenant!r} but the trace is for "
            f"{facts.trace_scope.tenant!r} (rule scope_violation)"
        )
    # 2. suppression must precede everything (Hard Rule 5)
    if facts.suppressed_at is None:
        raise SagaError(
            f"artifact {a.artifact_id} has no durable suppression journal entry; refusing to "
            "assign a status (rule not_suppressed, Hard Rule 5)"
        )

    def status(
        outcome: Outcome, level: VerifyLevel | None, reason: str, rule_id: str
    ) -> ArtifactStatus:
        return ArtifactStatus(
            artifact=a,
            suppressed_at=facts.suppressed_at or 0,
            reclaimed=facts.reclaimed,
            outcome=outcome,
            level=level,
            reason=reason,
            measurement=_m(facts),
            rule_id=rule_id,
        )

    if facts.dlq_error:
        return status(Outcome.UNVERIFIED, None, f"dlq: {facts.dlq_error}", "dlq")
    if facts.no_adapter:
        return status(
            Outcome.UNVERIFIED,
            None,
            f"no adapter for store {a.store!r}: suppression recorded in lineage only; "
            "configure the store in tombstone.yaml to erase and verify it",
            "no_adapter",
        )
    if facts.lineage_gap:
        return status(
            Outcome.UNVERIFIED,
            None,
            f"lineage-gap: store {a.store!r} holds entries with no lineage; erased under "
            "--accept-gaps, so untracked copies may remain",
            "lineage_gap",
        )
    # 3. third party
    if facts.is_third_party:
        return status(
            Outcome.NEEDS_HUMAN,
            None,
            "source document owned by another subject mentions this subject; not erased "
            "(no lineage edge; would require editing other subjects' data). Listed for human review.",
            "third_party",
        )
    # 4. logical
    if facts.logical_found:
        return status(
            Outcome.RESIDUAL,
            VerifyLevel.LOGICAL,
            f"still retrievable: {facts.logical_detail or 'a logical probe returned it'}",
            "logical_fail",
        )
    if facts.logical_found is None:
        return status(
            Outcome.UNVERIFIED, VerifyLevel.LOGICAL, "logical probe did not run", "logical_unprobed"
        )
    # 5. physical capability (before 6: managed stores must be UNVERIFIED, not VERIFIED)
    if not facts.physical_supported:
        return status(
            Outcome.UNVERIFIED,
            VerifyLevel.PHYSICAL,
            f"physical UNVERIFIED-managed: {facts.physical_reason or 'store cannot be physically verified'}",
            "physical_unsupported",
        )
    # 6. physical
    if facts.physical_found:
        return status(
            Outcome.RESIDUAL,
            VerifyLevel.PHYSICAL,
            f"bytes still present: {facts.physical_detail or 'pattern found in storage'}",
            "physical_fail",
        )
    if facts.physical_found is None:
        return status(
            Outcome.UNVERIFIED,
            VerifyLevel.PHYSICAL,
            "physical probe did not run",
            "physical_unprobed",
        )
    # 7. model
    if facts.model_applicable:
        has_mia = facts.mia_ci_low is not None and facts.mia_ci_high is not None
        if facts.model_probes_run == 0 and not has_mia:
            # nothing was decoded and no membership test ran: there is no evidence either way,
            # and "no evidence" is never a pass (Hard Rule 2)
            return status(
                Outcome.UNVERIFIED,
                VerifyLevel.MODEL,
                "model UNVERIFIED: no extraction prompts and no MIA reference set. Supply the "
                "subject's examples (they are snapshotted at suppress time) and set "
                "model.mia_reference in tombstone.yaml.",
                "model_unprobed",
            )
        canary_bad = (facts.canary_rate or 0.0) > 0.0
        mia_bad = (
            facts.mia_ci_low is not None
            and facts.mia_ci_high is not None
            and not (facts.mia_ci_low <= 0.5 <= facts.mia_ci_high)
        )
        if canary_bad or mia_bad:
            bits = []
            if facts.canary_extracted is not None and facts.canary_total is not None:
                bits.append(f"canary {facts.canary_extracted}/{facts.canary_total} extracted")
            if facts.mia_auc is not None:
                bits.append(
                    f"MIA AUC {facts.mia_auc:.2f} [CI {facts.mia_ci_low:.2f},{facts.mia_ci_high:.2f}]"
                )
            return status(
                Outcome.RESIDUAL,
                VerifyLevel.MODEL,
                "content still partially extractable: " + ", ".join(bits),
                "model_residual",
            )
    # 8. semantic (reported, never blocks — but it is RESIDUAL when measured above control)
    if facts.semantic_applicable and facts.drift is not None and facts.control is not None:
        eps = 1e-6  # floating-point noise is not drift
        ci_excludes_control = (
            facts.drift_ci_low is not None
            and facts.drift_ci_high is not None
            and not (facts.drift_ci_low - eps <= facts.control <= facts.drift_ci_high + eps)
            and facts.drift > facts.control + eps
        )
        if ci_excludes_control:
            return status(
                Outcome.RESIDUAL,
                VerifyLevel.SEMANTIC,
                f"retrieval-context drift {facts.drift:.4f} exceeds same-cluster control "
                f"{facts.control:.4f} (CI [{facts.drift_ci_low:.4f},{facts.drift_ci_high:.4f}]); "
                "a neighbourhood measurement, not proof the content is present",
                "semantic_residual",
            )
    # 9. verified at the highest applicable level
    # The store's verified level is physical (or model for adapters). Semantic drift is a
    # neighbourhood measurement reported alongside, never the level a store is verified at.
    level = VerifyLevel.MODEL if facts.model_applicable else VerifyLevel.PHYSICAL
    return status(Outcome.VERIFIED, level, "", "verified")
