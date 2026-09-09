"""2.4: pins detect silent changes; repin acknowledges them."""

from __future__ import annotations

from pathlib import Path

import pytest

from tombstone.errors import PinMismatch
from tombstone.model.pins import ManifestPin, ModelPin, StorePin, diff_pins
from tombstone.model.status import VerifyLevel

L, P, M = VerifyLevel.LOGICAL, VerifyLevel.PHYSICAL, VerifyLevel.MODEL


def sp(version="1.0", caps=(L, P), model="m", dims=384, backend="chroma"):
    return StorePin("chroma:kb", backend, version, frozenset(caps), model, dims)


CASES = [
    ("nothing changed", sp(), sp(), []),
    ("backend version bump", sp(), sp(version="1.1"), ["version"]),
    ("capabilities lost (managed instance)", sp(), sp(caps=(L,)), ["capabilities"]),
    ("capabilities gained", sp(caps=(L,)), sp(), ["capabilities"]),
    ("embedding model swapped", sp(), sp(model="bge"), ["embedding_model"]),
    ("dims changed", sp(), sp(dims=768), ["dims"]),
    ("backend kind changed", sp(), sp(backend="qdrant"), ["backend"]),
    (
        "adapter retrained",
        ModelPin("lora", "qwen", "aaaa", 16),
        ModelPin("lora", "qwen", "bbbb", 16),
        ["adapter_config_hash"],
    ),
    (
        "shard count changed",
        ModelPin("lora", "qwen", "aaaa", 16),
        ModelPin("lora", "qwen", "aaaa", 8),
        ["shard_count"],
    ),
    (
        "base model changed",
        ModelPin("lora", "qwen", "aaaa", 16),
        ModelPin("lora", "qwen-1.5b", "aaaa", 16),
        ["base_model"],
    ),
    (
        "manifest edited",
        ManifestPin("ft", "h1", 100),
        ManifestPin("ft", "h2", 100),
        ["manifest_hash"],
    ),
    (
        "manifest row count changed",
        ManifestPin("ft", "h1", 100),
        ManifestPin("ft", "h1", 99),
        ["example_count"],
    ),
    ("kind changed", sp(), ManifestPin("chroma:kb", "h", 1), ["kind"]),
]


@pytest.mark.parametrize("label,before,after,fields", CASES, ids=[c[0] for c in CASES])
def test_diff_pins(label, before, after, fields) -> None:
    deltas = diff_pins(before, after)
    assert [d.field for d in deltas] == fields, label
    for d in deltas:
        assert d.pin_name and d.before != d.after and str(d)


def test_pin_round_trip_dicts() -> None:
    from tombstone.model.pins import pin_from_dict

    for pin in (sp(), ModelPin("l", "b", "h", 4), ManifestPin("m", "h", 3)):
        assert pin_from_dict(pin.to_dict()) == pin


def test_live_pins_fire_on_silent_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A FAISS store is pinned; its index file is swapped for a different dims → PinMismatch."""
    from tests import _stores
    from tombstone.commands.init import run_init
    from tombstone.pins import check_pins, pin_all, repin, require_pins
    from tombstone.registry import Runtime

    monkeypatch.chdir(tmp_path)
    run_init(tmp_path)
    cfg = (tmp_path / "tombstone.yaml").read_text()
    cfg = cfg.replace(
        "stores:\n",
        'stores:\n  - { name: "faiss:kb", kind: faiss, path: ./kb.index, embedding: hash-embed-64 }\n',
    )
    (tmp_path / "tombstone.yaml").write_text(cfg)
    rt = Runtime.load(tmp_path / "tombstone.yaml")
    _stores.skip_unless("faiss")
    pins = pin_all(rt)
    assert "faiss:kb" in pins and pins["faiss:kb"].to_dict()["dims"] == 64
    assert check_pins(rt) == []
    require_pins(rt)
    rt.close()
    # silently rebuild the index with different dims (a "reindex with a new model" nobody told us about)
    from tombstone.stores.faiss import FaissStore

    for f in (
        tmp_path / "kb.index",
        tmp_path / "kb.index.meta.json",
        tmp_path / "kb.index.tombstones.json",
    ):
        f.unlink(missing_ok=True)
    FaissStore("faiss:kb", tmp_path / "kb.index", embedding_model="hash-embed-64", dims=128).close()
    rt = Runtime.load(tmp_path / "tombstone.yaml")
    deltas = check_pins(rt)
    assert [d.field for d in deltas] == ["dims"]
    with pytest.raises(PinMismatch) as ei:
        require_pins(rt)
    assert "dims" in str(ei.value) and "repin" in str(ei.value)
    with pytest.raises(PinMismatch, match="non-empty"):
        repin(rt, "   ")
    result = repin(rt, "re-embedded with a 128-dim model")
    assert [d.field for d in result["faiss:kb"]] == ["dims"]
    assert check_pins(rt) == []
    require_pins(rt)
    rt.close()
