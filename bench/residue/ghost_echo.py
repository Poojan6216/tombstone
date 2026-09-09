"""The *Ghost Echoes* drift protocol (arXiv 2608.20352, Trinity College, 2026) — credited.

We reproduce their measurement on our corpus with our numbers. This module runs the drift probe
from ``tombstone.verify.semantic`` around an erasure and, for task 7.5, estimates detection
accuracy of "was this subject ever here?" at query budgets 5/10/20/40 as the fraction of paired
comparisons in which the target's drift exceeds a same-cluster control's drift (the paper's
paired-comparison statistic), plus a leave-one-out threshold classifier over all pairs.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

from tombstone.stores._vector import VectorBackendBase
from tombstone.verify.semantic import DriftProbe


def measure_pair(
    store: VectorBackendBase, target_key: str, budget: int, erase: Callable[[], None], seed: int = 0
) -> dict[str, float] | None:
    """Record before, run ``erase`` (which removes the target), measure after."""
    probe = DriftProbe(store, budget=budget, seed=seed)
    if not probe.record_before(target_key):
        return None
    erase()
    return probe.after(target_key)


def _bootstrap_ci(
    pairs: Sequence[tuple[float, float]],
    stat: Callable[[Sequence[tuple[float, float]]], float],
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    """95% CI for a statistic over subjects, resampling subjects with replacement."""
    if len(pairs) < 2:
        return 0.0, 1.0
    rng = random.Random(seed)
    samples = sorted(stat([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(n_boot))
    return samples[int(0.025 * (n_boot - 1))], samples[int(0.975 * (n_boot - 1))]


def _paired(pairs: Sequence[tuple[float, float]]) -> float:
    return sum(1 for t, c in pairs if t > c) / max(1, len(pairs))


def _pooled_auc(pairs: Sequence[tuple[float, float]]) -> float:
    targets = [t for t, _ in pairs]
    controls = [c for _, c in pairs]
    if not targets or not controls:
        return 0.5
    wins = sum(1.0 if t > c else 0.5 if t == c else 0.0 for t in targets for c in controls)
    return wins / (len(targets) * len(controls))


def detection_accuracy(pairs: Sequence[tuple[float, float]]) -> dict[str, float]:
    """``pairs`` = (target_drift, control_drift) per subject.

    ``paired``: P(target > control), the paper's paired-comparison statistic; 50% is chance.

    Every rate carries a bootstrap 95% CI over subjects. At n=20 the interval is roughly ±0.20,
    which is what makes repeated runs of this experiment land on different point estimates; the
    index build is also not seeded (Chroma does not expose hnswlib's seed), so the neighbourhood
    itself differs slightly between runs. Quote the interval, never the point alone.

    ``pooled_auc``: P(a random subject's drift > a random subject's control drift), pooled across
    subjects. It answers the harder question — can an attacker who cannot construct a control for
    *this* subject still tell targets from controls? — and needs no threshold, so it is unbiased
    at this sample size.

    ``loo_threshold``: leave-one-out accuracy of a threshold classifier an attacker calibrates on
    the other subjects, picking the threshold *and the direction* (a rule that fires below the
    threshold is as usable as one that fires above it, and testing only "above" would understate
    the attacker). Choosing a threshold on n-1 overlapping points overfits, so this estimate is
    biased downward when there is no pooled signal and can read below chance; ``pooled_auc`` is
    the number to quote for that question.
    """
    if not pairs:
        return {"paired": 0.0, "loo_threshold": 0.0, "n": 0.0}
    paired = _paired(pairs)
    pooled_auc = _pooled_auc(pairs)
    paired_lo, paired_hi = _bootstrap_ci(pairs, _paired, seed=1)
    auc_lo, auc_hi = _bootstrap_ci(pairs, _pooled_auc, seed=2)
    values = [(t, 1) for t, _ in pairs] + [(c, 0) for _, c in pairs]
    correct = 0
    for i, (v, y) in enumerate(values):
        rest = values[:i] + values[i + 1 :]
        best: tuple[float, float, int] = (-1.0, 0.0, 1)  # (accuracy, threshold, direction)
        for thr in sorted({x for x, _ in rest}):
            for direction in (1, -1):
                acc = sum(
                    1 for x, yy in rest if ((x > thr) if direction == 1 else (x < thr)) == (yy == 1)
                ) / len(rest)
                if acc > best[0]:
                    best = (acc, thr, direction)
        _acc, thr, direction = best
        predicted = (v > thr) if direction == 1 else (v < thr)
        correct += int(predicted == (y == 1))
    return {
        "paired": paired,
        "paired_ci_low": paired_lo,
        "paired_ci_high": paired_hi,
        "pooled_auc": pooled_auc,
        "pooled_auc_ci_low": auc_lo,
        "pooled_auc_ci_high": auc_hi,
        "loo_threshold": correct / len(values),
        "n": float(len(pairs)),
    }


def summarize(results: Sequence[dict[str, float]]) -> dict[str, Any]:
    if not results:
        return {"n": 0}
    drifts = sorted(r["drift"] for r in results)
    controls = sorted(r["control"] for r in results)
    mid = len(drifts) // 2
    return {
        "n": len(results),
        "median_drift": drifts[mid],
        "median_control": controls[mid],
        "mean_drift": sum(drifts) / len(drifts),
        "mean_control": sum(controls) / len(controls),
        "above_control_rate": sum(1 for r in results if r["above_control"]) / len(results),
        "paired_target_gt_control": sum(1 for r in results if r["drift"] > r["control"])
        / len(results),
    }
