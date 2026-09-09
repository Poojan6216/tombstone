"""7.5 — Ghost Echoes after our best effort (B4): detection accuracy of "was this subject ever
here?" at query budgets 5, 10, 20, 40 (credit: arXiv 2608.20352 for the protocol)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import WORK, stable_sample
from adversarial._harness import Pipeline
from corpus.build import load_corpus
from residue.ghost_echo import detection_accuracy
from tombstone.commands.trace import run_trace
from tombstone.model.artifacts import ArtifactKind
from tombstone.verify.semantic import DriftProbe


def run(n_subjects: int) -> dict[str, Any]:
    docs = load_corpus()
    subjects = [f"S-{i:04d}" for i in range(1, n_subjects + 1)]
    public = stable_sample([d for d in docs if d.subject == "PUBLIC"], 600, 3)
    curve = []
    for budget in (5, 10, 20, 40):
        p = Pipeline(WORK / "attacks" / f"s75-b{budget}", backend="chroma")
        p.ingest([d for d in docs if d.subject in subjects] + public, capture=True)
        pairs = []
        for i, s in enumerate(subjects):
            t, _ = run_trace(p.rt, s, with_store_gaps=False)
            refs = [
                a for a in t.artifacts if a.kind is ArtifactKind.EMBED and a.store == p.store.name
            ]
            if not refs:
                continue
            probe = DriftProbe(p.store, budget=budget, seed=i)
            if not probe.record_before(refs[0].store_key):
                continue
            p.erase(s)
            r = probe.after(refs[0].store_key)
            if r:
                pairs.append((r["drift"], r["control"]))
        acc = detection_accuracy(pairs)
        curve.append({"budget": budget, **acc})
        p.close()
    b5 = curve[0]
    return {
        "id": "7.5",
        "name": "Ghost Echoes drift after full Tombstone erasure",
        "survives": "retrieval-context drift in the proximity graph (the layer we cannot close)",
        "rate": b5["paired"],
        "rate_text": "; ".join(
            f"budget {c['budget']}: paired {c['paired']:.1%}, threshold {c['loo_threshold']:.1%} (n={int(c['n'])})"
            for c in curve
        ),
        "mitigation": "none at the application layer; reported as RESIDUAL(semantic) when the CI excludes the control, never blocks, never claimed fixed",
        "detail": {"curve": curve},
    }
