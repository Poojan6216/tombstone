"""Verification levels, outcomes, per-artifact status, and the receipt."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from tombstone.model.artifacts import ArtifactRef, Scope, SubjectRef

SEMANTICS_VERSION = 1
"""Bumped whenever the status lattice or the probe semantics change. Recorded in every receipt."""


class VerifyLevel(StrEnum):
    LOGICAL = "logical"  # not retrievable by id / filter / top-k probe
    PHYSICAL = "physical"  # bytes not present in storage
    SEMANTIC = "semantic"  # retrieval-context drift at control level (measured, never certified)
    MODEL = "model"  # canaries not extractable; MIA at chance


class Outcome(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"  # could not check at this level; reason given
    RESIDUAL = "residual"  # checked, content still partially present; measurement given
    OUT_OF_SCOPE = "out_of_scope"  # layer the tool cannot reach; reason given
    NEEDS_HUMAN = "needs_human"  # third-party mention, legal hold, etc.


@dataclass(frozen=True, slots=True)
class ArtifactStatus:
    artifact: ArtifactRef
    suppressed_at: int  # journal seq. Hard Rule 5: always set before reclaim.
    reclaimed: bool
    outcome: Outcome
    level: VerifyLevel | None  # highest level VERIFIED, or the level RESIDUAL/UNVERIFIED hit
    reason: str  # human-readable, always non-empty for anything but VERIFIED
    measurement: Mapping[str, float] = field(default_factory=dict)
    rule_id: str = ""  # which lattice rule fired (see verify/levels.py)

    def __post_init__(self) -> None:
        if self.outcome is not Outcome.VERIFIED and not self.reason:
            raise ValueError(f"{self.outcome.value} status requires a non-empty reason")
        if self.outcome is Outcome.VERIFIED and self.level is None:
            raise ValueError("VERIFIED status requires a level")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "suppressed_at": self.suppressed_at,
            "reclaimed": self.reclaimed,
            "outcome": self.outcome.value,
            "level": self.level.value if self.level else None,
            "reason": self.reason,
            "measurement": {k: float(v) for k, v in sorted(self.measurement.items())},
            "rule_id": self.rule_id,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ArtifactStatus:
        return ArtifactStatus(
            artifact=ArtifactRef.from_dict(d["artifact"]),
            suppressed_at=int(d["suppressed_at"]),
            reclaimed=bool(d["reclaimed"]),
            outcome=Outcome(d["outcome"]),
            level=VerifyLevel(d["level"]) if d.get("level") else None,
            reason=str(d.get("reason", "")),
            measurement={str(k): float(v) for k, v in dict(d.get("measurement", {})).items()},
            rule_id=str(d.get("rule_id", "")),
        )


@dataclass(frozen=True, slots=True)
class Receipt:
    receipt_id: str
    trace_id: str
    subject: SubjectRef
    scope: Scope
    reason: str  # operator-supplied, e.g. "dsr-2026-0912"
    statuses: tuple[ArtifactStatus, ...]
    out_of_scope: tuple[str, ...]  # always non-empty. see Hard Rule 2.
    counts: Mapping[Outcome, int]
    journal_head: str  # hash of last journal record
    prev_receipt_hash: str
    signature: str  # Ed25519 over the canonical JSON of everything above
    semantics_version: int = SEMANTICS_VERSION
    created_ms: int = 0
    needs_human: tuple[ArtifactStatus, ...] = ()  # third-party sources, listed, never erased
    public_key: str = ""  # hex of the signing public key, for independent verification
    notes: tuple[str, ...] = ()  # e.g. lineage-gap acceptance, DLQ summary

    def __post_init__(self) -> None:
        if not self.out_of_scope:
            raise ValueError(
                "a receipt with an empty out_of_scope list cannot exist: there are always layers "
                "this tool cannot see (backups, WAL, provider-side logs). Declare them in "
                "tombstone.yaml under out_of_scope."
            )
        if any(not s.strip() for s in self.out_of_scope):
            raise ValueError("out_of_scope entries must be non-empty strings")

    def unsigned_payload(self) -> dict[str, Any]:
        """Everything the signature covers, in canonical form."""
        return {
            "receipt_id": self.receipt_id,
            "trace_id": self.trace_id,
            "subject": self.subject.hmac,
            "scope": self.scope.tenant,
            "reason": self.reason,
            "statuses": [s.to_dict() for s in self.statuses],
            "needs_human": [s.to_dict() for s in self.needs_human],
            "out_of_scope": list(self.out_of_scope),
            "counts": {k.value: int(v) for k, v in sorted(self.counts.items())},
            "journal_head": self.journal_head,
            "prev_receipt_hash": self.prev_receipt_hash,
            "semantics_version": self.semantics_version,
            "created_ms": self.created_ms,
            "notes": list(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        d = self.unsigned_payload()
        d["signature"] = self.signature
        d["public_key"] = self.public_key
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Receipt:
        return Receipt(
            receipt_id=str(d["receipt_id"]),
            trace_id=str(d["trace_id"]),
            subject=SubjectRef(str(d["subject"])),
            scope=Scope(str(d["scope"])),
            reason=str(d["reason"]),
            statuses=tuple(ArtifactStatus.from_dict(s) for s in d["statuses"]),
            out_of_scope=tuple(str(s) for s in d["out_of_scope"]),
            counts={Outcome(k): int(v) for k, v in dict(d["counts"]).items()},
            journal_head=str(d["journal_head"]),
            prev_receipt_hash=str(d["prev_receipt_hash"]),
            signature=str(d.get("signature", "")),
            semantics_version=int(d.get("semantics_version", SEMANTICS_VERSION)),
            created_ms=int(d.get("created_ms", 0)),
            needs_human=tuple(ArtifactStatus.from_dict(s) for s in d.get("needs_human", [])),
            public_key=str(d.get("public_key", "")),
            notes=tuple(str(n) for n in d.get("notes", [])),
        )

    def with_signature(self, signature: str, public_key: str) -> Receipt:
        return Receipt(
            receipt_id=self.receipt_id,
            trace_id=self.trace_id,
            subject=self.subject,
            scope=self.scope,
            reason=self.reason,
            statuses=self.statuses,
            out_of_scope=self.out_of_scope,
            counts=self.counts,
            journal_head=self.journal_head,
            prev_receipt_hash=self.prev_receipt_hash,
            signature=signature,
            semantics_version=self.semantics_version,
            created_ms=self.created_ms,
            needs_human=self.needs_human,
            public_key=public_key,
            notes=self.notes,
        )


def count_outcomes(statuses: tuple[ArtifactStatus, ...]) -> dict[Outcome, int]:
    counts: dict[Outcome, int] = dict.fromkeys(Outcome, 0)
    for s in statuses:
        counts[s.outcome] += 1
    return counts
