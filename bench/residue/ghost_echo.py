"""The *Ghost Echoes* drift protocol (arXiv 2608.20352, Trinity College, 2026) — credited.

We reproduce their measurement on our corpus with our numbers. This module runs the drift probe
from ``tombstone.verify.semantic`` around an erasure and, for task 7.5, estimates detection
accuracy of "was this subject ever here?" at query budgets 5/10/20/40 as the fraction of paired
comparisons in which the target's drift exceeds a same-cluster control's drift (the paper's
paired-comparison statistic), plus a leave-one-out threshold classifier over all pairs.
"""

from __future__ import annotations

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


def detection_accuracy(pairs: Sequence[tuple[float, float]]) -> dict[str, float]:
    """``pairs`` = (target_drift, control_drift) per subject.

    paired: P(target > control). loo_threshold: leave-one-out accuracy of a threshold classifier
    that labels a drift value 'deleted target' when above the threshold that best separates the
    remaining pairs (a budget-limited attacker who can calibrate on other subjects)."""
    if not pairs:
        return {"paired": 0.0, "loo_threshold": 0.0, "n": 0.0}
    paired = sum(1 for t, c in pairs if t > c) / len(pairs)
    values = [(t, 1) for t, _ in pairs] + [(c, 0) for _, c in pairs]
    correct = 0
    for i, (v, y) in enumerate(values):
        rest = values[:i] + values[i + 1 :]
        cands = sorted({x for x, _ in rest})
        best_thr, best_acc = 0.0, -1.0
        for thr in cands:
            acc = sum(1 for x, yy in rest if (x > thr) == (yy == 1)) / len(rest)
            if acc > best_acc:
                best_thr, best_acc = thr, acc
        correct += int((v > best_thr) == (y == 1))
    return {"paired": paired, "loo_threshold": correct / len(values), "n": float(len(pairs))}


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
