"""The Ghost Echoes detection metric must not understate what an attacker can do (7.5)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from residue.ghost_echo import detection_accuracy  # noqa: E402


def test_separable_pairs_are_detected() -> None:
    pairs = [(0.9 + i * 0.001, 0.1 + i * 0.001) for i in range(20)]
    r = detection_accuracy(pairs)
    assert r["paired"] == 1.0 and r["loo_threshold"] > 0.9


def test_no_signal_is_chance() -> None:
    r = detection_accuracy([(0.5, 0.5) for _ in range(20)])
    assert r["paired"] == 0.0 and r["loo_threshold"] == 0.5


def test_inverted_signal_is_still_a_signal() -> None:
    """An attacker calibrating on other subjects learns the direction; a one-sided rule would
    report 0% and understate the leak."""
    pairs = [(0.1 + i * 0.001, 0.9 + i * 0.001) for i in range(20)]
    r = detection_accuracy(pairs)
    assert r["paired"] == 0.0
    assert r["loo_threshold"] > 0.9, "the classifier must be allowed to pick the direction"


def test_overlapping_is_between() -> None:
    rng = random.Random(0)
    pairs = [(rng.gauss(0.55, 0.1), rng.gauss(0.5, 0.1)) for _ in range(20)]
    r = detection_accuracy(pairs)
    assert 0.4 <= r["paired"] <= 0.8 and 0.4 <= r["loo_threshold"] <= 0.9


def test_empty() -> None:
    assert detection_accuracy([])["n"] == 0.0
