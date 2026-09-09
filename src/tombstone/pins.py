"""Pin capture and checking (task 2.4).

At ``init``/first connect every store, model and manifest gets a pin. Before an erasure the
current state is re-observed and diffed; any delta without an explicit ``tombstone repin --reason``
raises ``PinMismatch`` naming the fields.
"""

from __future__ import annotations

from tombstone.errors import PinMismatch
from tombstone.model.pins import ManifestPin, ModelPin, Pin, PinDelta, StorePin, diff_pins
from tombstone.registry import Runtime
from tombstone.stores.base import ErasableStore


def observe_pin(name: str, store: ErasableStore) -> Pin:
    """What the store looks like right now."""
    kind = getattr(store, "kind", "store")
    if kind == "dataset":
        from tombstone.train.dataset import DatasetStore

        assert isinstance(store, DatasetStore)
        return store.pin()
    if kind == "adapter":
        from tombstone.stores.adapter import AdapterStore

        assert isinstance(store, AdapterStore)
        return (
            store.pin()
            if hasattr(store, "pin")
            else ModelPin(name, store.base_model, "", store.shards)
        )
    return StorePin(
        name=name,
        backend=kind,
        version=store.version(),
        capabilities=frozenset(store.capabilities),
        embedding_model=str(getattr(store, "embedding_model", "")),
        dims=int(getattr(store, "dims", 0) or 0),
    )


def pin_all(rt: Runtime, reason: str = "init") -> dict[str, Pin]:
    """Record a pin for every configured store that has none yet. Returns the current pins."""
    out: dict[str, Pin] = {}
    for name, store in rt.all_stores().items():
        current = rt.lineage.current_pin(name)
        if current is None:
            current = observe_pin(name, store)
            rt.lineage.put_pin(current, reason)
        out[name] = current
    return out


def check_pins(rt: Runtime, names: list[str] | None = None) -> list[PinDelta]:
    """Diff every pinned store against its live state. Unpinned stores are pinned now."""
    deltas: list[PinDelta] = []
    stores = rt.all_stores()
    for name, store in stores.items():
        if names is not None and name not in names:
            continue
        pinned = rt.lineage.current_pin(name)
        live = observe_pin(name, store)
        if pinned is None:
            rt.lineage.put_pin(live, "first connect")
            continue
        deltas.extend(diff_pins(pinned, live))
    return deltas


def require_pins(rt: Runtime, names: list[str] | None = None) -> None:
    deltas = check_pins(rt, names)
    if deltas:
        listing = "\n".join(f"  - {d}" for d in deltas)
        raise PinMismatch(
            "a store, model or manifest changed since it was pinned; refusing to erase until the "
            "change is acknowledged:\n"
            f"{listing}\n"
            "Review the change, then run: tombstone repin --reason '<why this changed>'"
        )


def repin(rt: Runtime, reason: str, names: list[str] | None = None) -> dict[str, list[PinDelta]]:
    """Acknowledge changes: record a fresh pin for each store (with the reason)."""
    if not reason.strip():
        raise PinMismatch("repin requires a non-empty --reason")
    out: dict[str, list[PinDelta]] = {}
    for name, store in rt.all_stores().items():
        if names is not None and name not in names:
            continue
        pinned = rt.lineage.current_pin(name)
        live = observe_pin(name, store)
        deltas = diff_pins(pinned, live) if pinned is not None else []
        if pinned is None or deltas:
            rt.lineage.put_pin(live, f"repin: {reason}")
        out[name] = deltas
    return out


__all__ = [
    "ManifestPin",
    "ModelPin",
    "PinDelta",
    "StorePin",
    "check_pins",
    "pin_all",
    "repin",
    "require_pins",
]
