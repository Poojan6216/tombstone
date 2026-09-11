"""The erasure saga: suppress(all) → reclaim(each) → verify(each) → receipt.

Every step is journaled *before* it runs (``step_begin``) and after (``step_end``); on restart
the saga reads the journal and continues from the first incomplete step. A failing store goes to
the dead-letter queue with the error and artifact ids; the saga continues with other stores and
the receipt marks those artifacts ``UNVERIFIED(dlq)``. No receipt is written while any step is
in progress (Hard Rule 6). The number of artifacts with a terminal status must equal the number
traced (Hard Rule 10).

Chaos hook: ``TOMBSTONE_CHAOS_KILL_AFTER=<n>`` sends SIGKILL to this process after the n-th
journal append, so the resume path can be tested exactly the way a crash would exercise it.
"""

from __future__ import annotations

import logging
import os
import signal
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tombstone.chain import Record
from tombstone.erase.journal import Journal
from tombstone.erase.reclaim import reclaim_in_store
from tombstone.erase.suppress import suppress_in_store, tombstone_in_lineage
from tombstone.errors import (
    ConfigError,
    LineageGapError,
    NotSupported,
    SagaError,
    ScopeViolation,
)
from tombstone.logging import get_logger, log
from tombstone.model.artifacts import ArtifactKind, ArtifactRef
from tombstone.model.lineage import Trace
from tombstone.model.status import Outcome, Receipt, VerifyLevel, count_outcomes
from tombstone.receipt.ledger import Ledger
from tombstone.receipt.render import ReceiptView, SemanticRow, StoreRow
from tombstone.receipt.sign import load_private_key, public_key_hex, sign_bytes
from tombstone.registry import Runtime
from tombstone.stores.base import ErasableStore
from tombstone.util import atomic_write_text, canonical_json, new_ulid, utc_ms
from tombstone.verify.levels import Facts, assign
from tombstone.verify.logical import build_probe_set

_log = get_logger("saga")


@dataclass
class SagaOptions:
    reason: str
    accept_gaps: bool = False
    reclaim: bool = True  # Hard Rule 5 test runs with reclaim disabled
    verify: bool = True
    semantic: bool = False  # Phase 5.3 drift measurement (slow); off by default in tests
    out_of_scope: tuple[str, ...] = ()


@dataclass
class _StoreGroup:
    name: str
    refs: list[ArtifactRef]
    store: ErasableStore | None
    error: str | None = None


@dataclass
class SagaResult:
    receipt: Receipt
    view: ReceiptView
    receipt_path: Path
    exit_code: int
    facts: dict[str, Facts] = field(default_factory=dict)


class ChaosJournal(Journal):
    """A Journal that kills the process after N appends when the chaos env var is set. It
    subclasses so every convenience writer (step_begin, probe, ...) counts too."""

    def __init__(self, path: Path, lock_timeout_s: float) -> None:
        super().__init__(path, lock_timeout_s)
        self.kill_after = int(os.environ.get("TOMBSTONE_CHAOS_KILL_AFTER", "0") or 0)
        self.appends = 0

    def append(self, type_: str, body: dict[str, Any]) -> Record:
        rec = super().append(type_, body)
        self.appends += 1
        if self.kill_after and self.appends >= self.kill_after:
            os.kill(os.getpid(), signal.SIGKILL)
        return rec


class Saga:
    def __init__(self, rt: Runtime, trace: Trace, opts: SagaOptions) -> None:
        self.rt = rt
        self.trace = trace
        self.opts = opts
        self.journal: Journal = ChaosJournal(rt.inst.journal_path, rt.cfg.erase.lock_timeout_s)
        self.ledger = Ledger(rt.inst.ledger_path, rt.cfg.erase.lock_timeout_s)
        self.saga_id = ""
        self.retry = False
        self._probe_records: list[Record] = []

    # --- preconditions -------------------------------------------------------------------------

    def _preflight(self) -> None:
        t = self.trace
        if t.scope != self.rt.scope:
            raise ScopeViolation(
                f"trace {t.trace_id} is for scope {t.scope.tenant!r}; this installation is "
                f"{self.rt.scope.tenant!r}"
            )
        if not t.artifacts:
            raise LineageGapError(
                f"trace {t.trace_id} has zero artifacts; refusing to erase nothing (Hard Rule 4)"
            )
        if t.gaps and not self.opts.accept_gaps:
            raise LineageGapError(
                f"trace {t.trace_id} has {len(t.gaps)} lineage gap(s); erase refuses unless "
                "--accept-gaps is passed (then the receipt records UNVERIFIED(lineage-gap) for the "
                "named stores):\n" + "\n".join(f"  ! {g}" for g in t.gaps)
            )
        # staleness: the graph (nodes/edges) must not have changed since the trace
        from tombstone.lineage.trace import trace as pure_trace

        snap = self.rt.lineage.snapshot(self.rt.scope, self._store_gaps_from_trace())
        fresh = pure_trace(t.subject, t.scope, snap)
        if fresh.snapshot_hash != t.snapshot_hash:
            raise SagaError(
                f"trace {t.trace_id} is stale: the lineage graph changed since it was taken "
                f"(snapshot {t.snapshot_hash[:12]} → {fresh.snapshot_hash[:12]}). Re-run "
                "`tombstone trace` and erase the new trace id (Hard Rule 10)."
            )
        from tombstone.pins import require_pins

        # Stores this saga has already begun to modify (a resumed run) are exempt: their change
        # is ours. Every other store must match its pin.
        touched: set[str] = set()
        existing = self.journal.saga_for_trace(t.trace_id)
        if existing and existing in self.journal.open_sagas():
            touched = {
                str(r.body.get("store"))
                for r in self.journal.records(existing)
                if r.type == Journal.STEP_BEGIN
            }
        require_pins(
            self.rt,
            sorted(
                {
                    a.store
                    for a in t.artifacts
                    if a.store in self._configured() and a.store not in touched
                }
            ),
        )

    def _store_gaps_from_trace(self) -> tuple[tuple[str, int], ...]:
        return tuple(self.trace.store_gaps)

    def _configured(self) -> set[str]:
        return {s.name for s in self.rt.cfg.stores}

    def _groups(self) -> list[_StoreGroup]:
        groups: dict[str, _StoreGroup] = {}
        for a in self.trace.artifacts:
            g = groups.get(a.store)
            if g is None:
                store: ErasableStore | None = None
                err: str | None = None
                if a.store in self._configured():
                    try:
                        store = self.rt.store(a.store)
                    except (ConfigError, Exception) as e:
                        err = f"{type(e).__name__}: {e}"
                g = _StoreGroup(a.store, [], store, err)
                groups[a.store] = g
            g.refs.append(a)
        return [groups[k] for k in sorted(groups)]

    # --- steps ---------------------------------------------------------------------------------

    def _done(self, step_id: str) -> Record | None:
        rec: Record | None = self.journal.completed_steps(self.saga_id).get(step_id)
        return rec

    def _run_step(
        self, step_id: str, phase: str, store: str, ids: Sequence[str], fn: Any
    ) -> tuple[bool, dict[str, Any], str | None]:
        """Journal begin → run → journal end. Returns (ok, result, error)."""
        prior = self._done(step_id)
        if prior is not None:
            return True, dict(prior.body.get("result", {})), None
        self.journal.step_begin(self.saga_id, step_id, phase, store, list(ids))
        try:
            result = fn() or {}
            noop = bool(result.pop("noop", False))
            self.journal.step_end(self.saga_id, step_id, ok=True, noop=noop, result=result)
            return True, result, None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.journal.step_end(self.saga_id, step_id, ok=False, error=err)
            log(
                _log, logging.ERROR, "step failed", saga_id=self.saga_id, step_id=step_id, error=err
            )
            return False, {}, err

    def run(self) -> SagaResult:
        self._preflight()
        t = self.trace
        existing = self.journal.saga_for_trace(t.trace_id)
        if existing and existing in self.journal.open_sagas():
            self.saga_id = existing
            log(_log, logging.INFO, "resuming saga", saga_id=self.saga_id, trace_id=t.trace_id)
        elif existing and existing not in self.journal.open_sagas() and self.retry:
            self.saga_id = new_ulid()
            self.journal.saga_start(
                self.saga_id,
                t.trace_id,
                t.subject.hmac,
                t.scope.tenant,
                self.opts.reason,
                len(t.artifacts),
                extra={
                    "accept_gaps": self.opts.accept_gaps,
                    "reclaim": self.opts.reclaim,
                    "retry_of": existing,
                },
            )
        elif existing and existing not in self.journal.open_sagas():
            raise SagaError(
                f"trace {t.trace_id} was already erased (saga {existing} has a receipt). "
                "Re-run `tombstone trace` for a fresh trace if there is new data."
            )
        else:
            self.saga_id = new_ulid()
            self.journal.saga_start(
                self.saga_id,
                t.trace_id,
                t.subject.hmac,
                t.scope.tenant,
                self.opts.reason,
                len(t.artifacts),
                extra={"accept_gaps": self.opts.accept_gaps, "reclaim": self.opts.reclaim},
            )
        groups = self._groups()
        dlq: dict[str, str] = {}  # store → error
        for g in groups:
            setter = getattr(g.store, "set_context", None)
            if callable(setter):
                setter(t)

        # ---- phase 0: semantic 'before' snapshot (Ghost Echoes) ---------------------------------
        if self.opts.semantic:
            from tombstone.verify.semantic import prepare_drift_before

            for g in groups:
                if g.store is not None and hasattr(g.store, "query"):
                    prepare_drift_before(g.store, g.refs, self.rt.cfg.erase.semantic_probe_budget)

        # ---- phase 1: suppress ----------------------------------------------------------------
        all_ids = [a.artifact_id for a in t.artifacts]
        ok, res, err = self._run_step(
            "suppress:lineage",
            "suppress",
            "lineage",
            all_ids,
            lambda: {
                "seq": tombstone_in_lineage(
                    self.rt.lineage, t.artifacts, t.trace_id, self.opts.reason
                )
            },
        )
        if not ok:
            raise SagaError(f"could not write lineage tombstones: {err}")
        suppressed_at = int(self.journal.completed_steps(self.saga_id)["suppress:lineage"].seq)
        suppressed_ok: set[str] = set()
        for g in groups:
            if g.store is None:
                continue  # no adapter: lineage tombstone is all we can do
            ok, _res, err = self._run_step(
                f"suppress:{g.name}",
                "suppress",
                g.name,
                [a.artifact_id for a in g.refs],
                lambda g=g: {"count": suppress_in_store(g.store, g.refs)},
            )
            if ok:
                suppressed_ok.update(a.artifact_id for a in g.refs)
                self._repin_after_reclaim(g)  # suppression may rewrite a manifest: our change
            else:
                dlq[g.name] = err or "suppress failed"
                self.journal.dlq(
                    self.saga_id,
                    f"suppress:{g.name}",
                    g.name,
                    [a.artifact_id for a in g.refs],
                    err or "",
                    0,
                )

        # ---- phase 1b: physical baselines for artifacts whose bytes are shared -----------------
        # A byte scan cannot tell this subject's copy of a vector from another subject's
        # byte-identical copy (boilerplate chunks). For those artifacts we count vector-pattern
        # matches now and call it residue only if the count does not drop after reclaim.
        for g in groups:
            if g.store is None or g.name in dlq or VerifyLevel.PHYSICAL not in g.store.capabilities:
                continue
            self._run_step(
                f"baseline:{g.name}",
                "baseline",
                g.name,
                [a.artifact_id for a in g.refs],
                lambda g=g: self._baseline_group(g),
            )

        # ---- phase 2: reclaim -----------------------------------------------------------------
        reclaim_results: dict[str, dict[str, Any]] = {}
        if self.opts.reclaim:
            for g in groups:
                if g.store is None or g.name in dlq:
                    continue
                ok, res, err = self._run_step(
                    f"reclaim:{g.name}",
                    "reclaim",
                    g.name,
                    [a.artifact_id for a in g.refs],
                    lambda g=g: _reclaim_payload(reclaim_in_store(g.store, g.refs)),
                )
                if ok:
                    reclaim_results[g.name] = res
                    self._repin_after_reclaim(g)
                else:
                    dlq[g.name] = err or "reclaim failed"
                    self.journal.dlq(
                        self.saga_id,
                        f"reclaim:{g.name}",
                        g.name,
                        [a.artifact_id for a in g.refs],
                        err or "",
                        0,
                    )

        # ---- phase 3: verify ------------------------------------------------------------------
        gap_stores = (
            {n for n, _ in self._store_gaps_from_trace()} if self.opts.accept_gaps else set()
        )
        facts: dict[str, Facts] = {}
        for g in groups:
            step_id = f"verify:{g.name}"
            if g.store is None:
                for a in g.refs:
                    facts[a.artifact_id] = Facts(a, t.scope, suppressed_at, no_adapter=True)
                continue
            if g.name in dlq:
                for a in g.refs:
                    facts[a.artifact_id] = Facts(a, t.scope, suppressed_at, dlq_error=dlq[g.name])
                continue
            prior = self._done(step_id)
            if prior is None and self.opts.verify:
                self._run_step(
                    step_id,
                    "verify",
                    g.name,
                    [a.artifact_id for a in g.refs],
                    lambda g=g: self._verify_group(g),
                )
            for a in g.refs:
                facts[a.artifact_id] = self._facts_from_journal(
                    a, suppressed_at, g, reclaim_results.get(g.name, {}), a.store in gap_stores
                )
        for a in t.third_party_hits:
            facts[a.artifact_id] = Facts(a, t.scope, suppressed_at, is_third_party=True)

        # ---- receipt --------------------------------------------------------------------------
        statuses = tuple(assign(facts[a.artifact_id]) for a in t.artifacts)
        needs_human = tuple(assign(facts[a.artifact_id]) for a in t.third_party_hits)
        if len(statuses) != len(t.artifacts):
            raise SagaError(
                f"erased {len(statuses)} artifacts but traced {len(t.artifacts)}; halting (Hard Rule 10)"
            )
        if self.journal.open_sagas() and any(
            r.type == Journal.STEP_BEGIN
            and r.body["step_id"] not in self.journal.completed_steps(self.saga_id)
            and not any(
                x.type == Journal.STEP_END and x.body["step_id"] == r.body["step_id"]
                for x in self.journal.records(self.saga_id)
            )
            for r in self.journal.records(self.saga_id)
        ):
            raise SagaError("a step is still in progress; no receipt is written (Hard Rule 6)")
        counts = count_outcomes(statuses)
        counts[Outcome.NEEDS_HUMAN] = counts.get(Outcome.NEEDS_HUMAN, 0) + len(needs_human)
        out_of_scope = tuple(self.opts.out_of_scope or self.rt.cfg.out_of_scope)
        notes: list[str] = []
        if self.opts.accept_gaps and t.gaps:
            notes.append(f"erased under --accept-gaps: {len(t.gaps)} lineage gap(s)")
        if dlq:
            notes.append("dead-lettered stores: " + ", ".join(sorted(dlq)))
        if not self.opts.reclaim:
            notes.append("reclaim disabled: suppression only")
        receipt = Receipt(
            receipt_id=new_ulid(),
            trace_id=t.trace_id,
            subject=t.subject,
            scope=t.scope,
            reason=self.opts.reason,
            statuses=statuses,
            out_of_scope=out_of_scope,
            counts=counts,
            journal_head=self.journal.head(),
            prev_receipt_hash=self.ledger.prev_receipt_hash(),
            signature="",
            created_ms=utc_ms(),
            needs_human=needs_human,
            notes=tuple(notes),
        )
        key = load_private_key(self.rt.inst.private_key_path)
        payload = canonical_json(receipt.unsigned_payload()).encode("utf-8")
        receipt = receipt.with_signature(sign_bytes(key, payload), public_key_hex(key))
        self.rt.inst.receipts_dir.mkdir(parents=True, exist_ok=True)
        path = self.rt.inst.receipts_dir / f"{receipt.receipt_id}.json"
        # the receipt is the deliverable; never let anything observe a half-written one
        atomic_write_text(path, canonical_json(receipt.to_dict()) + "\n")
        self.ledger.append(receipt)
        self.journal.saga_end(self.saga_id, receipt.receipt_id)
        self._purge_probes(statuses)
        for g in groups:
            fin = getattr(g.store, "finalize", None)
            if callable(fin):
                fin()
        exit_code = (
            0 if all(s.outcome is Outcome.VERIFIED for s in statuses) and not needs_human else 2
        )
        view = self._view(receipt, path, groups, reclaim_results, facts, dlq, exit_code)
        return SagaResult(receipt, view, path, exit_code, facts)

    # --- verification ----------------------------------------------------------------------------

    def _verify_group(self, g: _StoreGroup) -> dict[str, Any]:
        assert g.store is not None
        store = g.store
        dims = int(getattr(store, "dims", 0) or 0)
        k = self.rt.cfg.erase.logical_probe_k
        n_logical = 0
        n_physical = 0
        for a in g.refs:
            probes = build_probe_set(self.rt.lineage, a, dims, k)
            lr = store.probe_logical(a, probes)
            n_logical += 1
            self.journal.probe(
                self.saga_id,
                a.artifact_id,
                VerifyLevel.LOGICAL.value,
                "logical",
                lr.found if lr.probes_run else None,  # zero probes is not a pass
                {"probes_run": float(lr.probes_run)},
                detail=",".join(lr.found_by),
            )
            if VerifyLevel.PHYSICAL in store.capabilities:
                try:
                    pr = store.probe_physical(a)
                    n_physical += 1
                    self.journal.probe(
                        self.saga_id,
                        a.artifact_id,
                        VerifyLevel.PHYSICAL.value,
                        pr.method,
                        pr.found,
                        dict(pr.measurement),
                        detail=";".join(pr.locations) or pr.detail,
                    )
                except NotSupported as e:
                    self.journal.probe(
                        self.saga_id,
                        a.artifact_id,
                        VerifyLevel.PHYSICAL.value,
                        "unsupported",
                        None,
                        {},
                        detail=str(e),
                    )
            else:
                reason = getattr(
                    store, "physical_unsupported_reason", lambda: "no PHYSICAL capability"
                )()
                self.journal.probe(
                    self.saga_id,
                    a.artifact_id,
                    VerifyLevel.PHYSICAL.value,
                    "unsupported",
                    None,
                    {},
                    detail=str(reason),
                )
            model_probe = getattr(store, "probe_model", None)
            if a.kind is ArtifactKind.ADAPTER and callable(model_probe):
                mr = model_probe(a)
                self.journal.probe(
                    self.saga_id,
                    a.artifact_id,
                    VerifyLevel.MODEL.value,
                    mr.get("method", "model"),
                    bool(mr.get("found")),
                    {kk: float(v) for kk, v in mr.get("measurement", {}).items()},
                    detail=str(mr.get("detail", "")),
                )
        if self.opts.semantic and hasattr(store, "query"):
            from tombstone.verify.semantic import measure_drift_for_store

            sem = measure_drift_for_store(
                self.rt, store, g.refs, self.rt.cfg.erase.semantic_probe_budget
            )
            if sem is not None:
                for a in g.refs:
                    self.journal.probe(
                        self.saga_id,
                        a.artifact_id,
                        VerifyLevel.SEMANTIC.value,
                        "ghost-echo-drift",
                        bool(sem["above_control"]),
                        {kk: float(v) for kk, v in sem.items() if isinstance(v, int | float)},
                        detail="",
                    )
        return {"logical": n_logical, "physical": n_physical}

    def _repin_after_reclaim(self, g: _StoreGroup) -> None:
        """Our own reclaim legitimately changes a store (a rewritten manifest, a new adapter
        hash). Record the new pin so the change is not mistaken for a silent one."""
        from tombstone.model.pins import diff_pins
        from tombstone.pins import observe_pin

        assert g.store is not None
        live = observe_pin(g.name, g.store)
        pinned = self.rt.lineage.current_pin(g.name)
        if pinned is None or diff_pins(pinned, live):
            self.rt.lineage.put_pin(live, f"reclaim by saga {self.saga_id}")

    def _baseline_group(self, g: _StoreGroup) -> dict[str, Any]:
        assert g.store is not None
        n = 0
        for a in g.refs:
            dupes = self.rt.lineage.live_duplicates(
                a.store,
                a.embedding_fingerprint,
                a.content_hash,
                a.artifact_id,
                erasing=[x.artifact_id for x in self.trace.artifacts],
            )
            if dupes <= 0:
                # Record the zero explicitly rather than skipping. A missing baseline record
                # and a recorded zero both decide the same way today, but they do not resume
                # the same way: the first baseline record for an artifact wins on resume, so a
                # recorded value is inherited from the killed attempt, while a missing one is
                # recomputed later under whatever state the resumed run finds. No physical
                # probe is needed — with the duplicate rule decided on the lineage graph alone,
                # the byte count carried no decision weight, and the probe was the expensive
                # half of this step.
                self.journal.probe(
                    self.saga_id,
                    a.artifact_id,
                    VerifyLevel.PHYSICAL.value,
                    "baseline",
                    False,
                    {"live_duplicates": 0.0},
                    detail="no other live artifact holds these bytes",
                )
                continue
            try:
                pr = g.store.probe_physical(a)
            except NotSupported:
                continue
            n += 1
            self.journal.probe(
                self.saga_id,
                a.artifact_id,
                VerifyLevel.PHYSICAL.value,
                "baseline",
                pr.found,
                {
                    "content_matches": _content_matches(pr.measurement),
                    "live_duplicates": float(dupes),
                },
                detail=";".join(pr.locations),
            )
        return {"baselines": n}

    def _facts_from_journal(
        self, a: ArtifactRef, suppressed_at: int, g: _StoreGroup, reclaim: dict[str, Any], gap: bool
    ) -> Facts:
        store = g.store
        assert store is not None
        # Probe records for this artifact from this saga, plus the *earliest* baseline recorded
        # by any saga of the same trace (a retry must compare against the count measured before
        # the first reclaim, when this artifact's own copy was still present).
        sagas_for_trace = {
            r.body["saga_id"]
            for r in self.journal.records()
            if r.type == Journal.SAGA_START and r.body.get("trace_id") == self.trace.trace_id
        }
        probes: list[Record] = []
        baseline_seen = False
        for r in self.journal.records():
            if r.type != Journal.PROBE or r.body.get("artifact_id") != a.artifact_id:
                continue
            if r.body.get("probe") == "baseline":
                if not baseline_seen and r.body.get("saga_id") in sagas_for_trace:
                    probes.append(r)
                    baseline_seen = True
                continue
            if r.body.get("saga_id") == self.saga_id:
                probes.append(r)
        f: dict[str, Any] = {
            "artifact": a,
            "trace_scope": self.trace.scope,
            "suppressed_at": suppressed_at,
            "reclaimed": bool(reclaim),
            "reclaim_method": str(reclaim.get("method", "")),
            "lineage_gap": gap,
            "physical_supported": VerifyLevel.PHYSICAL in store.capabilities,
            "physical_reason": getattr(store, "physical_unsupported_reason", lambda: "")()
            if VerifyLevel.PHYSICAL not in store.capabilities
            else "",
            "model_applicable": a.kind is ArtifactKind.ADAPTER
            and VerifyLevel.MODEL in store.capabilities,
            "semantic_applicable": False,
        }
        extra: dict[str, float] = {}
        for r in probes:
            b = r.body
            if b["level"] == "logical":
                f["logical_found"] = bool(b["found"])
                f["logical_detail"] = f"found by {b['detail']}" if b["detail"] else ""
            elif b["level"] == "physical":
                if b["probe"] == "unsupported":
                    f["physical_supported"] = False
                    f["physical_reason"] = b.get("detail", "")
                elif b["probe"] == "baseline":
                    m = b.get("measurement", {})
                    extra["baseline_content_matches"] = float(m.get("content_matches", 0.0))
                    extra["live_duplicates"] = float(m.get("live_duplicates", 0.0))
                else:
                    m = b.get("measurement", {})
                    id_hits = float(m.get("matches_artifact_id", m.get("matches_id", 0.0)))
                    content_hits = _content_matches(m)
                    if id_hits > 0:
                        found = True
                        detail = f"{b['probe']}: this artifact's own record bytes present ({b['detail']})"
                    elif content_hits > 0 and extra.get("live_duplicates", 0.0) > 0:
                        # Other live artifacts hold byte-identical content, so a byte scan cannot
                        # say whose copy it found, and this artifact's own record is already gone
                        # (id_hits == 0 above). The outcome is decided by that fact about the
                        # lineage graph alone — never by comparing byte counts before and after.
                        #
                        # Every count-based version of this rule was timing-sensitive. Chroma
                        # flushes segments from a background thread, so the same probe returns
                        # different numbers depending on when it runs, and a saga killed and
                        # resumed measured at a different moment and reached a different verdict:
                        # the same three artifacts flipped between VERIFIED and UNVERIFIED at
                        # three separate kill points. A receipt that depends on when the machine
                        # died is not a receipt (Hard Rule 9), and no amount of waiting makes a
                        # measured count into a stable one on a machine you do not own.
                        #
                        # The cost is deliberate: shared boilerplate now reports UNVERIFIED rather
                        # than VERIFIED. That is the honest answer — we cannot attribute those
                        # bytes — and it is what the independent verifier already says about the
                        # same artifacts.
                        found = None
                        f["physical_supported"] = False
                        f["physical_reason"] = (
                            f"duplicate content: {int(extra['live_duplicates'])} other live "
                            "artifact(s) hold byte-identical content; this artifact's own "
                            "record is absent but its content bytes cannot be attributed"
                        )
                        detail = f"{b['probe']}: content bytes shared with other live artifacts"
                    elif content_hits > 0:
                        found = True
                        detail = f"{b['probe']}: {b['detail']}"
                    else:
                        found = False
                        detail = f"{b['probe']}: {b['detail']}" if b["detail"] else b["probe"]
                    f["physical_found"] = found
                    f["physical_detail"] = detail
                    for kk, v in m.items():
                        if kk in {"dead_tuples", "matches"}:
                            extra[kk] = float(v)
            elif b["level"] == "model":
                m = b.get("measurement", {})
                f["model_applicable"] = True
                f["canary_rate"] = m.get("canary_rate")
                f["canary_extracted"] = (
                    int(m["canary_extracted"]) if "canary_extracted" in m else None
                )
                f["canary_total"] = int(m["canary_total"]) if "canary_total" in m else None
                f["model_probes_run"] = int(m.get("canary_total", 0))
                f["mia_auc"] = m.get("mia_auc")
                f["mia_ci_low"] = m.get("mia_ci_low")
                f["mia_ci_high"] = m.get("mia_ci_high")
            elif b["level"] == "semantic":
                m = b.get("measurement", {})
                f["semantic_applicable"] = True
                f["drift"] = m.get("drift")
                f["control"] = m.get("control")
                f["drift_ci_low"] = m.get("drift_ci_low")
                f["drift_ci_high"] = m.get("drift_ci_high")
                f["control_ci_low"] = m.get("control_ci_low")
                f["control_ci_high"] = m.get("control_ci_high")
                f["query_budget"] = int(m["query_budget"]) if "query_budget" in m else None
        f["measurement_extra"] = extra
        return Facts(**f)

    def _purge_probes(self, statuses: Sequence[Any]) -> None:
        verified = [s.artifact.artifact_id for s in statuses if s.outcome is Outcome.VERIFIED]
        if not verified:
            return
        with self.rt.lineage.tx():
            for i in range(0, len(verified), 500):
                batch = verified[i : i + 500]
                marks = ",".join("?" for _ in batch)
                self.rt.lineage._exec(f"DELETE FROM probes WHERE artifact_id IN ({marks})", batch)

    # --- view ------------------------------------------------------------------------------------

    def _view(
        self,
        receipt: Receipt,
        path: Path,
        groups: list[_StoreGroup],
        reclaim_results: dict[str, dict[str, Any]],
        facts: dict[str, Facts],
        dlq: dict[str, str],
        exit_code: int,
    ) -> ReceiptView:
        by_id = {s.artifact.artifact_id: s for s in receipt.statuses}
        rows: list[StoreRow] = []
        semantic: list[SemanticRow] = []
        for g in groups:
            sts = [by_id[a.artifact_id] for a in g.refs]
            n = len(sts)
            n_ok = sum(1 for s in sts if s.outcome is Outcome.VERIFIED)
            worst = _worst(sts)
            method = str(reclaim_results.get(g.name, {}).get("method", "")) or (
                "suppressed only"
                if not self.opts.reclaim
                else ("—" if g.store is not None else "no adapter")
            )
            if g.name in dlq:
                method = "FAILED → dlq"
            level = worst.level.value if worst.level else "-"
            label = _label(worst)
            detail = _detail(worst, n_ok, n)
            note = ""
            if worst.outcome is Outcome.UNVERIFIED and worst.rule_id == "physical_unsupported":
                note = "→ run VACUUM FULL / REINDEX from an owner role, then `tombstone verify --receipt`"
                if g.store is not None and getattr(g.store, "kind", "") != "pgvector":
                    note = "→ grant this process access to the persisted index, then `tombstone verify --receipt`"
            elif worst.outcome is Outcome.RESIDUAL and worst.level is VerifyLevel.MODEL:
                note = "→ content still partially extractable. Options: full retrain, or a RESIDUAL receipt"
            rows.append(StoreRow(g.name, method[:34], level, label, detail, note))
            f0 = facts.get(g.refs[0].artifact_id)
            if (
                f0 is not None
                and f0.semantic_applicable
                and f0.drift is not None
                and f0.control is not None
            ):
                verdict = (
                    "above control"
                    if worst.rule_id == "semantic_residual"
                    or any(s.rule_id == "semantic_residual" for s in sts)
                    else "at control level"
                )
                semantic.append(SemanticRow(g.name, f0.drift, f0.control, verdict))
        return ReceiptView(
            receipt_path=str(path),
            suppressed=len(receipt.statuses),
            total=len(receipt.statuses),
            rows=tuple(rows),
            semantic=tuple(semantic),
            needs_human=tuple(s.artifact.artifact_id for s in receipt.needs_human),
            chain_ok=True,
            signed=bool(receipt.signature),
            exit_code=exit_code,
            dlq=tuple(sorted(dlq)),
        )


def _reclaim_payload(r: Any) -> dict[str, Any]:
    return {
        "noop": bool(r.noop),
        "method": r.method,
        "measurement": {k: float(v) for k, v in dict(r.measurement).items()},
        "detail": r.detail,
    }


_RANK = {
    Outcome.RESIDUAL: 0,
    Outcome.UNVERIFIED: 1,
    Outcome.NEEDS_HUMAN: 2,
    Outcome.OUT_OF_SCOPE: 3,
    Outcome.VERIFIED: 4,
}


def _worst(statuses: Sequence[Any]) -> Any:
    return sorted(statuses, key=lambda s: (_RANK[s.outcome], s.artifact.artifact_id))[0]


def _label(s: Any) -> str:
    if s.outcome is Outcome.UNVERIFIED and s.rule_id == "physical_unsupported":
        return "UNVERIFIED-managed"
    if s.outcome is Outcome.UNVERIFIED and s.rule_id == "lineage_gap":
        return "UNVERIFIED-lineage-gap"
    if s.outcome is Outcome.UNVERIFIED and s.rule_id == "dlq":
        return "UNVERIFIED-dlq"
    return str(s.outcome.value).upper()


def _detail(s: Any, n_ok: int, n: int) -> str:
    m = s.measurement
    if s.outcome is Outcome.VERIFIED:
        if s.level is VerifyLevel.PHYSICAL:
            if "dead_tuples" in m:
                return f"(pgstattuple dead={int(m['dead_tuples'])}, bytes absent, {n_ok}/{n})"
            return f"(bytes absent, {n_ok}/{n})"
        if s.level is VerifyLevel.MODEL:
            ce, ct = m.get("canary_extracted"), m.get("canary_total")
            auc = m.get("mia_auc")
            bits = []
            if ce is not None and ct is not None:
                bits.append(f"canary {int(ce)}/{int(ct)}")
            if auc is not None:
                bits.append(
                    f"MIA AUC {auc:.2f} [CI {m.get('mia_ci_low', 0):.2f},{m.get('mia_ci_high', 0):.2f}]"
                )
            return " ".join(bits)
        return f"({n_ok}/{n})"
    if s.outcome is Outcome.RESIDUAL:
        return f"({s.reason[:70]})"
    return f"({s.reason[:90]})"


def _content_matches(m: Any) -> float:
    """Sum of pattern matches that identify content (vector bytes, content hash) rather than
    the artifact's own id — the part that byte-identical copies from other subjects share."""
    total = 0.0
    for k, v in dict(m).items():
        if k.startswith("matches_") and k not in {
            "matches_artifact_id",
            "matches_id",
            "matches_cache_key",
        }:
            total += float(v)
    return total
