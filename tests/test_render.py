"""2.5: renderers perform no computation — output equals the precomputed struct, field by field."""

from __future__ import annotations

from tests._graphs import graph, node
from tombstone.commands.trace import render_status, render_trace
from tombstone.lineage.trace import trace
from tombstone.model.artifacts import Scope, SubjectRef


def test_render_trace_golden() -> None:
    snap = graph(
        [
            node("S1"),
            node("S2", subject="s2"),
            node("C1"),
            node("E1", store="chroma:kb-v2"),
            node("E2", store="pgvector:kb-v1"),
            node("K1", store="semantic-cache"),
        ],
        [
            ("S1", "C1", "chunk"),
            ("C1", "E1", "embed:a"),
            ("C1", "E2", "embed:b"),
            ("C1", "K1", "cache:semantic"),
        ],
        mentions=[("S2", "s1")],
        store_gaps=(("faiss:kb", 20),),
        tombstoned={"E1"},
    )
    t = trace(SubjectRef("s1"), Scope("default"), snap)
    out = render_trace(t)
    expected = "\n".join(
        [
            f"trace:    {t.trace_id}",
            "subject:  hmac:s1…s1   (raw id never logged)",
            "scope:    default",
            f"snapshot: sha256:{t.snapshot_hash[:16]}…",
            "artifacts descending from this subject: 5",
            "",
            "  chroma:kb-v2                    1   embed×1   already-tombstoned:1",
            "  pgvector:kb-v1                  1   embed×1",
            "  semantic-cache                  1   cache×1",
            "  store-chunk                     1   chunk×1",
            "  store-source                    1   source×1",
            "",
            "third-party mentions (NEEDS_HUMAN, never erased): 1",
            "  source S2 in store-source",
            "",
            "gaps: 1 — erase will refuse without --accept-gaps",
            "  ! store faiss:kb: ~20 entries with no lineage node",
            "",
            f"next: tombstone erase --trace {t.trace_id} --reason <dsr-id> --confirm",
        ]
    )
    assert out == expected


def test_render_status_golden() -> None:
    payload = {
        "scope": "default",
        "lineage": {
            "backend": "sqlite",
            "nodes_by_kind": {"chunk": 10, "embed": 20, "source": 2},
            "subjects": 2,
        },
        "stores": [
            {
                "name": "chroma:kb-v2",
                "kind": "chroma",
                "entries": 20,
                "lineage_nodes": 20,
                "coverage": 1.0,
                "unlineaged_estimate": 0,
                "capabilities": ["logical", "physical"],
            },
            {
                "name": "faiss:kb",
                "kind": "faiss",
                "entries": 40,
                "lineage_nodes": 20,
                "coverage": 0.5,
                "unlineaged_estimate": 20,
                "capabilities": ["logical"],
            },
        ],
        "gaps": ["faiss:kb: ~20 of 40 entries have no lineage node (10/20 in sample)"],
        "pins": {"chroma:kb-v2": {"kind": "store"}},
        "journal": {"records": 3, "open_sagas": ["S1"]},
        "dlq_depth": 1,
        "receipts": 4,
    }
    out = render_status(payload)
    assert out == "\n".join(
        [
            "scope: default   lineage: sqlite   subjects: 2",
            "nodes: chunk=10, embed=20, source=2",
            "",
            "  store                        kind            entries  lineage  coverage  capabilities",
            "  chroma:kb-v2                 chroma               20       20    100.0%  logical,physical",
            "  faiss:kb                     faiss                40       20     50.0%  logical",
            "",
            "  ! faiss:kb: ~20 of 40 entries have no lineage node (10/20 in sample)",
            "",
            "pins: 1  chroma:kb-v2(store)",
            "journal: 3 records, open sagas: 1   dlq depth: 1   receipts: 4",
        ]
    )
