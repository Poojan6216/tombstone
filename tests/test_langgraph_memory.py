"""8.2: LangGraph-style memory entries become MEMORY nodes and erase through the same saga."""

from __future__ import annotations

from pathlib import Path

import pytest

from tombstone.commands.erase import run_erase
from tombstone.commands.trace import run_trace
from tombstone.integrations.langgraph import TombstoneMemoryStore
from tombstone.model.artifacts import ArtifactKind
from tombstone.stores.memory import MemoryStore


def test_memory_round_trip_with_erasure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tombstone.commands.init import run_init
    from tombstone.registry import Runtime

    monkeypatch.chdir(tmp_path)
    run_init(tmp_path)
    cfg = tmp_path / "tombstone.yaml"
    cfg.write_text(
        cfg.read_text().replace(
            "stores:\n", 'stores:\n  - { name: "agent-memory", kind: memory }\n'
        )
    )
    rt = Runtime.shared(cfg)
    mem = rt.store("agent-memory")
    assert isinstance(mem, MemoryStore)
    ms = TombstoneMemoryStore(mem, rt.capture(), rt.pepper())
    ms.put(("users", "u1"), "prefs", {"likes": "tea", "note": "membership KX7"}, subject_id="S-U1")
    ms.put(("users", "u1"), "history", {"last": "refund"}, subject_id="S-U1")
    ms.put(("users", "u2"), "prefs", {"likes": "coffee"}, subject_id="S-U2")
    assert ms.get(("users", "u1"), "prefs").value["likes"] == "tea"  # type: ignore[union-attr]
    assert len(ms.search(("users",))) == 3 and len(ms.search(("users",), query="tea")) == 1
    t, _ = run_trace(rt, "S-U1")
    kinds = {a.kind for a in t.artifacts}
    assert (
        ArtifactKind.MEMORY in kinds
        and sum(1 for a in t.artifacts if a.kind is ArtifactKind.MEMORY) == 2
    )
    code, text, data = run_erase(rt, t.trace_id, "dsr-mem", confirm=True)
    mem_rows = [s for s in data["statuses"] if s["artifact"]["kind"] == "memory"]
    assert mem_rows and all(s["outcome"] == "verified" for s in mem_rows), text
    assert ms.get(("users", "u1"), "prefs") is None and ms.get(("users", "u1"), "history") is None
    assert ms.get(("users", "u2"), "prefs") is not None
    # the app's own delete is recorded, not treated as an erasure
    ms.delete(("users", "u2"), "prefs")
    t2, _ = run_trace(rt, "S-U2")
    assert t2.already_tombstoned
    rt.close()
