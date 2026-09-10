"""The independent verifier (task 5.6): no lineage db, no pepper.

Given a receipt file (and optionally a public key PEM and the ledger), it checks the signature
and the chain, then re-runs every probe the receipt claims against the live stores using only
the artifact refs in the receipt. Credit: ``forgetlayer`` frames this as "a store grading its
own homework convinces no auditor"; this is that mode.

Without the lineage db there is no probe table, so the logical probe uses the id lookup, the
metadata filter, and the padded-fingerprint top-k. Content bytes shared with other live records
(boilerplate) are attributed by asking the store itself for live records with identical content
(an O(n) scan; fine for an audit). Documented in verification-levels.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tombstone.config import load_config
from tombstone.errors import ChainBroken, NotSupported, SignatureInvalid
from tombstone.model.status import Outcome, Receipt, VerifyLevel
from tombstone.receipt.ledger import Ledger
from tombstone.receipt.sign import (
    load_public_key,
    public_key_from_hex,
    public_key_hex,
    verify_bytes,
)
from tombstone.registry import Runtime
from tombstone.stores.base import ProbeSet
from tombstone.util import canonical_json
from tombstone.verify.logical import padded_fingerprint

_ID_KEYS = {"matches_artifact_id", "matches_id", "matches_cache_key"}


def _check_signature(receipt: Receipt, public_key_pem: Path | None) -> tuple[bool, str]:
    """(verified?, what that verification is worth).

    A receipt carries the public key that signed it, and ``public_key`` is deliberately *not*
    inside ``unsigned_payload()`` — it cannot be, since the signature is over the payload. So
    checking a receipt against its own embedded key proves the bytes were not altered after
    signing; it proves nothing about *who* signed. Anyone can write a receipt, sign it with a key
    they generated, and it verifies. Only a key obtained out of band — the operator's
    ``.tombstone/keys/public.pem`` — turns that into a statement about origin, and the caller is
    told which of the two they got.
    """
    payload = canonical_json(receipt.unsigned_payload()).encode("utf-8")
    if public_key_pem is not None:
        key = load_public_key(public_key_pem)
        if public_key_hex(key) != receipt.public_key:
            return False, f"the receipt was signed by a different key than {public_key_pem}"
        trust = f"verified against {public_key_pem}"
    else:
        key = public_key_from_hex(receipt.public_key)
        trust = (
            "key taken from the receipt itself: this shows the receipt was not altered after "
            "signing, not who signed it. Pass --public-key .tombstone/keys/public.pem to check "
            "it came from your installation"
        )
    try:
        verify_bytes(key, payload, receipt.signature)
        return True, trust
    except SignatureInvalid:
        return False, trust


def _check_chain(receipt: Receipt, ledger: Path) -> tuple[bool, str]:
    try:
        led = Ledger(ledger)
        n = led.verify()
    except ChainBroken as e:
        return False, f"chain: BROKEN ({e})"
    records = led.records()
    idx = next(
        (i for i, r in enumerate(records) if r.body.get("receipt_id") == receipt.receipt_id), None
    )
    if idx is None:
        return False, f"chain: receipt not found in ledger ({n} records)"
    prev_receipts = [r for r in records[:idx] if r.type == Ledger.RECEIPT]
    expected_prev = prev_receipts[-1].hash if prev_receipts else "0" * 64
    if receipt.prev_receipt_hash != expected_prev:
        return False, "chain: prev_receipt_hash does not match the ledger"
    # A receipt verified against its own embedded key says nothing about origin, but the ledger
    # gives a second opinion for free: every receipt in one installation's ledger should be signed
    # by that installation's key. One that is not was signed by someone else, whatever its own
    # signature says.
    others = {
        str(r.body.get("public_key"))
        for r in records
        if r.type == Ledger.RECEIPT and r.body.get("public_key")
    } - {receipt.public_key}
    if others:
        return (
            False,
            f"chain: receipt at index {idx} is signed by a key that signs none of the other "
            f"{len(prev_receipts)} receipt(s) in this ledger ({len(others)} other key(s) present)",
        )
    return True, f"chain: OK ({n} records, receipt at index {idx})"


def _physical_observation(store: Any, a: Any) -> tuple[bool | None, str]:
    """(bytes present?, description). None when the store cannot be checked here."""
    try:
        pr = store.probe_physical(a)
    except NotSupported:
        return None, "physical not checkable here"
    m = dict(pr.measurement)
    has_patterns = any(k.startswith("matches_") for k in m)
    id_hits = (
        float(sum(float(v) for k, v in m.items() if k in _ID_KEYS))
        if has_patterns
        else float(pr.found)
    )
    content_hits = sum(
        float(v) for k, v in m.items() if k.startswith("matches_") and k not in _ID_KEYS
    )
    if id_hits > 0:
        return True, f"physical RESIDUAL: own record bytes present ({';'.join(pr.locations)})"
    if content_hits > 0:
        dup_fn = getattr(store, "live_content_duplicates", None)
        dupes = int(dup_fn(a)) if callable(dup_fn) else 0
        if dupes > 0:
            return (
                False,
                f"physical absent (content bytes attributable to {dupes} live record(s) with identical content)",
            )
        return True, f"physical RESIDUAL ({';'.join(pr.locations)})"
    return False, "physical absent"


def verify_receipt_independently(
    receipt_path: Path,
    public_key_pem: Path | None = None,
    ledger: Path | None = None,
    config: str | Path | None = None,
) -> dict[str, Any]:
    receipt = Receipt.from_dict(json.loads(receipt_path.read_text(encoding="utf-8")))
    lines: list[str] = [
        f"receipt {receipt.receipt_id}  trace {receipt.trace_id}  subject {receipt.subject.short}"
    ]
    ok, trust = _check_signature(receipt, public_key_pem)
    if not ok:
        lines.append(f"signature: INVALID ({trust})")
    elif public_key_pem is not None:
        lines.append(f"signature: ed25519 OK ({trust})")
    else:
        lines.append(f"signature: ed25519 self-asserted — {trust}")
    if ledger is not None:
        chain_ok, msg = _check_chain(receipt, ledger)
        lines.append(msg)
        ok = ok and chain_ok
    else:
        lines.append("chain: not checked (pass --ledger to check)")
    cfg, cfg_path = load_config(config)
    rt = Runtime(cfg, cfg_path)
    rows: list[dict[str, Any]] = []
    try:
        configured = {x.name for x in cfg.stores}
        for s in receipt.statuses:
            a = s.artifact
            claimed = s.outcome
            if a.store not in configured:
                rows.append(
                    {
                        "artifact_id": a.artifact_id,
                        "store": a.store,
                        "claimed": claimed.value,
                        "observed": "no adapter",
                        "agree": claimed is not Outcome.VERIFIED,
                    }
                )
                continue
            store = rt.store(a.store)
            dims = int(getattr(store, "dims", 0) or 0)
            vectors = (
                (padded_fingerprint(a.embedding_fingerprint, dims),)
                if a.embedding_fingerprint and dims
                else ()
            )
            lr = store.probe_logical(a, ProbeSet(a.artifact_id, vectors, cfg.erase.logical_probe_k))
            present = bool(lr.found)
            observed = "logical RESIDUAL" if present else "logical absent"
            if not present and VerifyLevel.PHYSICAL in store.capabilities:
                phys, desc = _physical_observation(store, a)
                observed = desc
                if phys and s.rule_id == "physical_unsupported":
                    observed += " (receipt said UNVERIFIED)"
                present = bool(phys)
            agree = claimed is not Outcome.VERIFIED or not present
            rows.append(
                {
                    "artifact_id": a.artifact_id,
                    "store": a.store,
                    "claimed": claimed.value,
                    "observed": observed,
                    "agree": agree,
                }
            )
    finally:
        rt.close()
    disagreements = [r for r in rows if not r["agree"]]
    if disagreements:
        ok = False
    lines.append(
        f"probes: {len(rows)} artifacts re-checked, {len(disagreements)} disagree with the receipt"
    )
    for r in disagreements:
        lines.append(
            f"  RESIDUAL {r['store']} {r['artifact_id']}: receipt says {r['claimed']}, live store: {r['observed']}"
        )
    lines.append("verdict: " + ("receipt reproduced" if ok else "receipt NOT reproduced"))
    return {"ok": ok, "receipt_id": receipt.receipt_id, "rows": rows, "text": "\n".join(lines)}
