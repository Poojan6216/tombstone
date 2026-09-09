"""7.4 — third-party mentions with no ``mentions`` edge: the app never extracted entities.
Tombstone does not search for them by design (Hard Rule 3). Non-zero by construction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import WORK
from adversarial._harness import Pipeline
from corpus.build import Doc, load_corpus


def run(n_subjects: int) -> dict[str, Any]:
    docs = load_corpus()
    subjects = [f"S-{i:04d}" for i in range(1, n_subjects + 1)]
    canaries = {d.subject: d.canary for d in docs if d.subject in subjects and d.canary}
    # other subjects' documents that quote the target's canary (no mentions edge recorded)
    extra: list[Doc] = []
    for i, (s, can) in enumerate(sorted(canaries.items())):
        other = f"S-{(i % 50) + 101:04d}"
        extra.append(
            Doc(
                f"{other}-quote-{s}",
                other,
                f"support/{other}/quote-{s}.txt",
                f"Case note. The agent copied the earlier record: {can.sentence} Follow-up scheduled.",
                (),
                None,
                0,
            )
        )
    base = [d for d in docs if d.subject in subjects or d.subject in {e.subject for e in extra}]
    p = Pipeline(WORK / "attacks" / "s74")
    p.ingest(base + extra, capture=True)
    survived = 0
    listed = 0
    for s, can in canaries.items():
        _code, data, _t = p.erase(s)
        survived += int(p.canary_hits(can.token) > 0)
        listed += len(data.get("needs_human", []))
    p.close()
    n = len(canaries)
    return {
        "id": "7.4",
        "name": "third-party mentions without an edge",
        "survives": "the subject's data quoted inside other subjects' documents",
        "rate": survived / max(1, n),
        "rate_text": f"{survived}/{n} subjects' canaries survive in other subjects' documents (no mentions edge → {listed} NEEDS_HUMAN rows listed)",
        "mitigation": "none inside Tombstone by design: searching other subjects' data by similarity would over-delete their data (Hard Rule 3). The app must record mentions=[...] at ingest; then the documents are listed NEEDS_HUMAN for review (docs/threat-model.md).",
        "detail": {"subjects": n, "survived": survived},
    }
