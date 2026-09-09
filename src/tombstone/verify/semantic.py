"""Semantic residue: the *Ghost Echoes* protocol (arXiv 2608.20352, Trinity College, 2026).

Credit: this is their measurement, reproduced on our corpus. We do not claim it as ours.

Protocol, as implemented here:
  1. Before deletion, induce ``budget`` queries around the target (its own vector plus
     perturbations toward its nearest neighbours) and record each query's Top-K (K=5) result
     set and centroid — the target included, because that is what a user would have retrieved.
  2. Delete the target (the erasure does this). Re-run the same queries and measure how far the
     Top-K centroid moved. That is the drift.
  3. Control, the paper's "same-cluster" arm: for each query, count how many Top-K slots the
     subject occupied, and build the counterfactual in which that many of the *nearest
     non-subject* entries had been deleted instead. Matching on slots vacated (not on vectors
     deleted) is what makes the two arms comparable; matching on vector count lets the control
     vacate more of the retrieved set than the erasure does, and the target then loses to its
     own control by construction.
  4. Report drift, control, a bootstrap CI over the queries, the query budget, and the mean
     number of Top-K slots the subject held.

RESIDUAL(semantic) when the drift CI excludes the control and drift > control. This level is
reported, never blocks, and is never called proof that the content is present: it is a
measurement of the neighbourhood.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

from tombstone.model.artifacts import ArtifactRef
from tombstone.stores._vector import VectorBackendBase, cosine
from tombstone.verify.logical import build_probe_set

K = 5


def _centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    d = len(vectors[0])
    out = [0.0] * d
    for v in vectors:
        for i in range(d):
            out[i] += v[i]
    return [x / len(vectors) for x in out]


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    d = 1.0 - cosine(a, b)
    return 0.0 if d < 1e-9 else d  # cosine of identical vectors rounds to 1 ± 1e-16


def _perturb(v: Sequence[float], toward: Sequence[float], alpha: float) -> list[float]:
    out = [(1 - alpha) * x + alpha * y for x, y in zip(v, toward, strict=False)]
    n = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / n for x in out]


def induce_queries(
    store: VectorBackendBase, target_vec: Sequence[float], budget: int, seed: int = 0
) -> list[list[float]]:
    """``budget`` queries around the target: the target itself, then perturbations toward its
    nearest neighbours (deterministic given the seed)."""
    rng = random.Random(seed)
    neigh = [h for h in store.query(target_vec, K + 1)]
    vecs = []
    for h in neigh:
        v = store._vector_of(h.key)
        if v is not None:
            vecs.append(v)
    out: list[list[float]] = [list(target_vec)]
    while len(out) < budget:
        if vecs:
            toward = vecs[rng.randrange(len(vecs))]
            out.append(_perturb(target_vec, toward, rng.uniform(0.2, 0.6)))
        else:
            noise = [rng.gauss(0, 0.05) for _ in target_vec]
            out.append(_perturb(target_vec, noise, 0.5))
    return out


def topk_keys(
    store: VectorBackendBase, query: Sequence[float], k: int, exclude: set[str]
) -> list[str]:
    return [h.key for h in store.query(query, k + len(exclude)) if h.key not in exclude][:k]


def topk_vectors(
    store: VectorBackendBase, query: Sequence[float], k: int, exclude: set[str]
) -> list[list[float]]:
    hits = [h for h in store.query(query, k + len(exclude)) if h.key not in exclude][:k]
    out = []
    for h in hits:
        v = store._vector_of(h.key)
        if v is not None:
            out.append(v)
    return out


def bootstrap_mean(
    values: Sequence[float], n_boot: int = 300, seed: int = 0
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    mean = sum(values) / len(values)
    samples = []
    for _ in range(n_boot):
        s = [values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(s) / len(s))
    samples.sort()
    return mean, samples[int(0.025 * (n_boot - 1))], samples[int(0.975 * (n_boot - 1))]


class DriftProbe:
    """Records the neighbourhood before an erasure; ``after()`` measures how far it moved.

    The comparison is the paper's paired one, and both arms must be non-degenerate:

    * **drift** — the Top-K centroid for an induced query *before* the deletion (the target
      included, exactly what a user would have retrieved) against the same query *after* it.
    * **control** — the same "before" centroid against a counterfactual in which the same number
      of Top-K *slots* had been vacated by deleting nearest non-subject entries instead. That is
      what any deletion of that size does to the neighbourhood; drift above it is the subject's
      own trace.

    Two ways to get a meaningless answer here, both of which this code hit first and now guards
    against: excluding the target from the "before" set makes both arms identical and every
    measurement exactly zero; matching the control on vectors deleted rather than slots vacated
    makes the control the larger perturbation and the target lose by construction.
    """

    def __init__(self, store: VectorBackendBase, budget: int = 5, seed: int = 0) -> None:
        self.store = store
        self.budget = budget
        self.seed = seed
        self.before: dict[str, dict[str, Any]] = {}

    def record_before(self, key: str, deleted_keys: Sequence[str] | None = None) -> bool:
        """``key`` anchors the queries; ``deleted_keys`` is everything the erasure will remove.

        The control is matched **per query, by slots vacated**: for each induced query, count how
        many of its Top-K the subject occupies, then build the counterfactual in which that many
        of the *nearest non-subject* entries had been deleted instead. Matching on the number of
        deleted vectors instead would let the control vacate more Top-K slots than the erasure
        does, and the target would lose to its own control by construction.
        """
        vec = self.store._vector_of(key)
        if vec is None:
            return False
        targets = set(deleted_keys or [key])
        queries = induce_queries(self.store, vec, self.budget, self.seed)
        before: list[list[float]] = []
        control: list[list[float]] = []
        vacated: list[int] = []
        for q in queries:
            keys_before = topk_keys(self.store, q, K, set())
            before.append(
                _centroid([v for v in (self.store._vector_of(k) for k in keys_before) if v])
            )
            m = sum(1 for k in keys_before if k in targets)
            vacated.append(m)
            # rank order, not set order: the control must be identical on every run (Hard Rule 9)
            drop = {k for k in [k for k in keys_before if k not in targets][:m]}
            control.append(_centroid(topk_vectors(self.store, q, K, drop)))
        self.before[key] = {
            "queries": queries,
            "centroids": before,
            "control_centroids": control,
            "slots_vacated": vacated,
            "targets": sorted(targets),
        }
        return True

    def after(self, key: str) -> dict[str, float] | None:
        b = self.before.get(key)
        if b is None:
            return None
        drifts: list[float] = []
        controls: list[float] = []
        for q, c_before, c_control in zip(
            b["queries"], b["centroids"], b["control_centroids"], strict=True
        ):
            after_c = _centroid(topk_vectors(self.store, q, K, set()))
            drifts.append(_dist(c_before, after_c))
            controls.append(_dist(c_before, c_control))
        slots = b.get("slots_vacated") or [0]
        d_mean, d_lo, d_hi = bootstrap_mean(drifts, seed=self.seed)
        c_mean, c_lo, c_hi = bootstrap_mean(controls, seed=self.seed + 1)
        above = d_mean > c_mean and not (d_lo <= c_mean <= d_hi)
        return {
            "drift": d_mean,
            "drift_ci_low": d_lo,
            "drift_ci_high": d_hi,
            "control": c_mean,
            "control_ci_low": c_lo,
            "control_ci_high": c_hi,
            "query_budget": float(self.budget),
            "slots_vacated_mean": sum(slots) / max(1, len(slots)),
            "above_control": 1.0 if above else 0.0,
        }


def measure_drift_for_store(
    rt: Any, store: Any, refs: Sequence[ArtifactRef], budget: int
) -> dict[str, Any] | None:
    """Saga hook: needs a 'before' snapshot taken by the saga at suppress time (kept on the
    store as ``_drift_probe``). Returns the store-level mean over the refs, or None."""
    probe: DriftProbe | None = getattr(store, "_drift_probe", None)
    if probe is None or not isinstance(store, VectorBackendBase):
        return None
    results = [r for r in (probe.after(ref.store_key) for ref in refs) if r is not None]
    if not results:
        return None
    n = float(len(results))
    out = {k: sum(r[k] for r in results) / n for k in results[0]}
    out["above_control"] = 1.0 if any(r["above_control"] for r in results) else 0.0
    return out


def prepare_drift_before(
    store: Any, refs: Sequence[ArtifactRef], budget: int, seed: int = 0
) -> int:
    """Called by the saga before suppression when --semantic is on."""
    if not isinstance(store, VectorBackendBase):
        return 0
    probe = DriftProbe(store, budget, seed)
    keys = [r.store_key for r in refs]
    n = sum(1 for r in refs if probe.record_before(r.store_key, keys))
    store._drift_probe = probe  # type: ignore[attr-defined]
    _ = build_probe_set  # keep the import meaningful for readers: probes reuse the same vectors
    return n
