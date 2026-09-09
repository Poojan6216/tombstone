"""Semantic residue (Ghost Echoes protocol). Implemented in Phase 5.3; until then the saga
records nothing at the semantic level (it is never claimed)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tombstone.model.artifacts import ArtifactRef


def measure_drift_for_store(
    rt: Any, store: Any, refs: Sequence[ArtifactRef], budget: int
) -> dict[str, Any] | None:
    return None
