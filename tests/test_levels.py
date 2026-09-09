"""5.4: the lattice, receipts, signatures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tombstone.errors import SagaError, ScopeViolation, SignatureInvalid
from tombstone.model.artifacts import ArtifactKind, ArtifactRef, Scope, SubjectRef
from tombstone.model.status import ArtifactStatus, Outcome, Receipt, VerifyLevel, count_outcomes
from tombstone.receipt.sign import (
    generate_keypair,
    load_private_key,
    load_public_key,
    public_key_hex,
    sign_bytes,
    verify_bytes,
)
from tombstone.util import canonical_json
from tombstone.verify.levels import Facts, assign

D = Scope("default")


def ref(
    kind: ArtifactKind = ArtifactKind.EMBED, store: str = "chroma:kb", scope: str = "default"
) -> ArtifactRef:
    return ArtifactRef(
        "01A",
        kind,
        store,
        "k",
        Scope(scope),
        "h" * 64,
        "00" * 128 if kind is ArtifactKind.EMBED else None,
    )


def facts(**kw) -> Facts:
    base = dict(
        artifact=ref(),
        trace_scope=D,
        suppressed_at=3,
        reclaimed=True,
        logical_found=False,
        physical_supported=True,
        physical_found=False,
    )
    base.update(kw)
    return Facts(**base)


GOLDEN = [
    ("verified physical", facts(), Outcome.VERIFIED, VerifyLevel.PHYSICAL, "verified"),
    ("dlq", facts(dlq_error="disk on fire"), Outcome.UNVERIFIED, None, "dlq"),
    ("no adapter", facts(no_adapter=True), Outcome.UNVERIFIED, None, "no_adapter"),
    ("lineage gap", facts(lineage_gap=True), Outcome.UNVERIFIED, None, "lineage_gap"),
    (
        "third party",
        facts(artifact=ref(ArtifactKind.SOURCE), is_third_party=True),
        Outcome.NEEDS_HUMAN,
        None,
        "third_party",
    ),
    (
        "logical residual",
        facts(logical_found=True, logical_detail="found by id"),
        Outcome.RESIDUAL,
        VerifyLevel.LOGICAL,
        "logical_fail",
    ),
    (
        "logical unprobed",
        facts(logical_found=None),
        Outcome.UNVERIFIED,
        VerifyLevel.LOGICAL,
        "logical_unprobed",
    ),
    (
        "managed store",
        facts(physical_supported=False, physical_reason="not owner"),
        Outcome.UNVERIFIED,
        VerifyLevel.PHYSICAL,
        "physical_unsupported",
    ),
    (
        "physical residual",
        facts(physical_found=True, physical_detail="chroma.sqlite3"),
        Outcome.RESIDUAL,
        VerifyLevel.PHYSICAL,
        "physical_fail",
    ),
    (
        "physical unprobed",
        facts(physical_found=None),
        Outcome.UNVERIFIED,
        VerifyLevel.PHYSICAL,
        "physical_unprobed",
    ),
    (
        "model residual canary",
        facts(
            artifact=ref(ArtifactKind.ADAPTER, "lora"),
            model_applicable=True,
            canary_rate=0.2,
            canary_extracted=1,
            canary_total=5,
            mia_auc=0.63,
            mia_ci_low=0.58,
            mia_ci_high=0.68,
        ),
        Outcome.RESIDUAL,
        VerifyLevel.MODEL,
        "model_residual",
    ),
    (
        "model residual mia only",
        facts(
            artifact=ref(ArtifactKind.ADAPTER, "lora"),
            model_applicable=True,
            canary_rate=0.0,
            canary_extracted=0,
            canary_total=5,
            mia_auc=0.7,
            mia_ci_low=0.6,
            mia_ci_high=0.8,
        ),
        Outcome.RESIDUAL,
        VerifyLevel.MODEL,
        "model_residual",
    ),
    (
        "model verified",
        facts(
            artifact=ref(ArtifactKind.ADAPTER, "lora"),
            model_applicable=True,
            canary_rate=0.0,
            canary_extracted=0,
            canary_total=5,
            mia_auc=0.51,
            mia_ci_low=0.47,
            mia_ci_high=0.55,
        ),
        Outcome.VERIFIED,
        VerifyLevel.MODEL,
        "verified",
    ),
    (
        "semantic residual",
        facts(
            semantic_applicable=True,
            drift=0.15,
            control=0.03,
            drift_ci_low=0.12,
            drift_ci_high=0.18,
        ),
        Outcome.RESIDUAL,
        VerifyLevel.SEMANTIC,
        "semantic_residual",
    ),
    (
        "semantic at control",
        facts(
            semantic_applicable=True,
            drift=0.041,
            control=0.038,
            drift_ci_low=0.03,
            drift_ci_high=0.05,
        ),
        Outcome.VERIFIED,
        VerifyLevel.SEMANTIC,
        "verified",
    ),
    # precedence: logical beats physical beats model beats semantic
    (
        "precedence logical>physical",
        facts(logical_found=True, physical_found=True),
        Outcome.RESIDUAL,
        VerifyLevel.LOGICAL,
        "logical_fail",
    ),
    (
        "precedence managed>physical fail",
        facts(physical_supported=False, physical_found=True),
        Outcome.UNVERIFIED,
        VerifyLevel.PHYSICAL,
        "physical_unsupported",
    ),
    (
        "precedence dlq>everything",
        facts(dlq_error="x", logical_found=True),
        Outcome.UNVERIFIED,
        None,
        "dlq",
    ),
]


@pytest.mark.parametrize("label,f,outcome,level,rule", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_lattice_golden(label, f, outcome, level, rule) -> None:
    s = assign(f)
    assert (s.outcome, s.level, s.rule_id) == (outcome, level, rule), label
    if s.outcome is not Outcome.VERIFIED:
        assert s.reason
    assert s.suppressed_at == 3


def test_lattice_raises_on_scope_and_suppression() -> None:
    with pytest.raises(ScopeViolation):
        assign(facts(artifact=ref(scope="other")))
    with pytest.raises(SagaError, match="Hard Rule 5"):
        assign(facts(suppressed_at=None))


def test_status_requires_reason_unless_verified() -> None:
    with pytest.raises(ValueError):
        ArtifactStatus(ref(), 1, True, Outcome.UNVERIFIED, None, "")
    with pytest.raises(ValueError):
        ArtifactStatus(ref(), 1, True, Outcome.VERIFIED, None, "")


def _receipt(out_of_scope=("backups",), statuses=()) -> Receipt:
    return Receipt(
        receipt_id="R1",
        trace_id="T1",
        subject=SubjectRef("s" * 64),
        scope=D,
        reason="dsr-1",
        statuses=tuple(statuses),
        out_of_scope=tuple(out_of_scope),
        counts=count_outcomes(tuple(statuses)),
        journal_head="j" * 64,
        prev_receipt_hash="0" * 64,
        signature="",
    )


def test_receipt_with_empty_out_of_scope_cannot_exist() -> None:
    with pytest.raises(ValueError, match="out_of_scope"):
        _receipt(out_of_scope=())
    with pytest.raises(ValueError):
        _receipt(out_of_scope=("  ",))


def test_receipt_round_trip_and_signature(tmp_path: Path) -> None:
    generate_keypair(tmp_path / "k.key", tmp_path / "k.pub")
    key = load_private_key(tmp_path / "k.key")
    r = _receipt(statuses=[assign(facts())])
    payload = canonical_json(r.unsigned_payload()).encode()
    signed = r.with_signature(sign_bytes(key, payload), public_key_hex(key))
    d = signed.to_dict()
    back = Receipt.from_dict(json.loads(json.dumps(d)))
    assert back == signed
    pub = load_public_key(tmp_path / "k.pub")
    verify_bytes(pub, canonical_json(back.unsigned_payload()).encode(), back.signature)
    # any byte change breaks it
    tampered = json.loads(json.dumps(d))
    tampered["reason"] = "dsr-2"
    t = Receipt.from_dict(tampered)
    with pytest.raises(SignatureInvalid):
        verify_bytes(pub, canonical_json(t.unsigned_payload()).encode(), t.signature)
    tampered = json.loads(json.dumps(d))
    tampered["statuses"][0]["outcome"] = "unverified"
    tampered["statuses"][0]["reason"] = "x"
    t = Receipt.from_dict(tampered)
    with pytest.raises(SignatureInvalid):
        verify_bytes(pub, canonical_json(t.unsigned_payload()).encode(), t.signature)


def test_banned_words_absent_from_lattice_and_render_outputs() -> None:
    from tombstone.receipt.render import ReceiptView, render_receipt

    r = _receipt(statuses=[assign(facts())])
    text = render_receipt(r, ReceiptView("r.json", 1, 1, ()))
    low = text.lower()
    for word in ("certified", "guaranteed", "complete erasure", "gdpr compliant"):
        assert word not in low
    assert "not a legal instrument" in low
