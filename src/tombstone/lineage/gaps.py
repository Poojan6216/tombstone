"""Lineage gap detection (Hard Rule 4).

Three kinds of gap:
  1. a node with no path back to a SOURCE (computed purely in ``trace``);
  2. a registered store with no capture activity in the scope;
  3. store contents with no lineage node — data ingested before Tombstone was installed, or
     written around the wrapper. Detected by sampling the store's keys and looking them up.

This module does the I/O (2, 3) and hands the result to the pure trace as ``store_gaps``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from tombstone.lineage.store import LineageStore
from tombstone.model.artifacts import Scope
from tombstone.stores.base import ErasableStore


@dataclass(frozen=True, slots=True)
class StoreCoverage:
    store: str
    total: int  # entries in the store
    sampled: int
    unlineaged_in_sample: int

    @property
    def unlineaged_estimate(self) -> int:
        if self.sampled == 0:
            return 0
        if self.sampled >= self.total:
            return self.unlineaged_in_sample
        return round(self.total * self.unlineaged_in_sample / self.sampled)

    @property
    def coverage(self) -> float:
        if self.total == 0:
            return 1.0
        return max(0.0, 1.0 - self.unlineaged_estimate / self.total)


@dataclass(frozen=True, slots=True)
class GapReport:
    coverage: tuple[StoreCoverage, ...]
    stores_without_capture: tuple[str, ...]
    messages: tuple[str, ...] = field(default=())

    def store_gaps(self) -> tuple[tuple[str, int], ...]:
        """(store, unlineaged count) per store; -1 when a store could not be counted."""
        out = {c.store: c.unlineaged_estimate for c in self.coverage if c.unlineaged_estimate > 0}
        for s in self.stores_without_capture:
            out.setdefault(s, -1)
        return tuple(sorted(out.items()))

    @property
    def has_gaps(self) -> bool:
        return bool(self.store_gaps())


def detect_gaps(
    lineage: LineageStore,
    scope: Scope,
    stores: Mapping[str, ErasableStore],
    sample: int = 500,
) -> GapReport:
    counts = lineage.counts_by_store(scope)
    registered = [name for name, _ in lineage.registered_stores(scope)]
    coverage: list[StoreCoverage] = []
    messages: list[str] = []
    no_capture: list[str] = []
    for name in sorted(set(registered) | set(stores)):
        store = stores.get(name)
        if store is None:
            continue
        try:
            total = store.count()
            keys = store.sample_keys(sample)
        except Exception as e:
            messages.append(f"{name}: could not sample store ({type(e).__name__}: {e})")
            continue
        present = lineage.store_keys_present(name, keys) if keys else set()
        # Caches key their lineage rows as <key>@<subject16>; match on the key prefix.
        if store.kind in {"cache_exact", "cache_semantic"}:
            present = _prefix_present(lineage, name, keys)
        missing = len([k for k in keys if k not in present])
        cov = StoreCoverage(name, total, len(keys), missing)
        coverage.append(cov)
        if total > 0 and counts.get(name, 0) == 0:
            no_capture.append(name)
            messages.append(
                f"{name}: {total} entries in the store, none with lineage — ingested before "
                "capture was enabled, or written around the wrapper"
            )
        elif cov.unlineaged_estimate > 0:
            messages.append(
                f"{name}: ~{cov.unlineaged_estimate} of {total} entries have no lineage node "
                f"({cov.unlineaged_in_sample}/{cov.sampled} in sample)"
            )
    return GapReport(tuple(coverage), tuple(sorted(no_capture)), tuple(messages))


def _prefix_present(lineage: LineageStore, store: str, keys: list[str]) -> set[str]:
    present: set[str] = set()
    for k in keys:
        rows = lineage._exec(
            "SELECT 1 FROM nodes WHERE store = ? AND store_key LIKE ? LIMIT 1", (store, f"{k}@%")
        ).fetchall()
        if rows:
            present.add(k)
    return present
