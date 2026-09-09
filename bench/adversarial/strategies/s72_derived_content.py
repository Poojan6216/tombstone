"""7.2 — derived content without an edge: the app summarises a document and stores the summary
as a new document, unstamped or stamped under another subject. Measured with and without the
``derived_from`` mitigation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import WORK
from adversarial._harness import Pipeline
from corpus.build import Doc, load_corpus
from tombstone.lineage.stamp import source_artifact_id
from tombstone.model.artifacts import Scope, SubjectRef


def _summary(d: Doc) -> Doc:
    # an extractive "summary" that keeps the canary sentence (the worst case for residue)
    sents = d.text.split(". ")
    keep = [s for s in sents if d.canary and d.canary.token in s] or sents[:1]
    return Doc(
        d.doc_id + "-summary",
        "APP",
        d.source + ".summary",
        "Summary: " + ". ".join(keep),
        (),
        None,
        d.cluster,
    )


def run(n_subjects: int) -> dict[str, Any]:
    docs = load_corpus()
    subjects = [f"S-{i:04d}" for i in range(1, n_subjects + 1)]
    canary_docs = [d for d in docs if d.subject in subjects and d.canary]
    results = {}
    for mode in ("unstamped", "derived_from"):
        p = Pipeline(WORK / "attacks" / f"s72-{mode}")
        p.ingest(canary_docs, capture=True)
        summaries = [_summary(d) for d in canary_docs]
        if mode == "unstamped":
            # stored as a new doc under the app's own "subject" — Tombstone sees no edge
            p.ingest(summaries, capture=True, subject_override={s.doc_id: "APP" for s in summaries})
        else:
            derived = {}
            for d, s in zip(canary_docs, summaries, strict=True):
                src_id = source_artifact_id(
                    Scope("default"), SubjectRef.from_raw(d.subject, p.pepper), d.source
                )
                derived[s.doc_id] = src_id
            p.ingest(
                summaries,
                capture=True,
                subject_override={
                    s.doc_id: d.subject for d, s in zip(canary_docs, summaries, strict=True)
                },
                derived_from=derived,
            )
        survived = 0
        for d in canary_docs:
            p.erase(d.subject)
            survived += int(p.canary_hits(d.canary.token) > 0)
        results[mode] = {
            "subjects": len(canary_docs),
            "canary_survived": survived,
            "rate": survived / max(1, len(canary_docs)),
        }
        p.close()
    return {
        "id": "7.2",
        "name": "derived content without an edge",
        "survives": "LLM summaries stored as new, unstamped documents",
        "rate": results["unstamped"]["rate"],
        "rate_text": f"unstamped summaries: {results['unstamped']['canary_survived']}/{results['unstamped']['subjects']} canaries survive; with derived_from stamping: {results['derived_from']['canary_survived']}/{results['derived_from']['subjects']}",
        "mitigation": "stamp derived documents with derived_from=<source artifact id> (the app must do it; Tombstone cannot see the edge otherwise)",
        "detail": results,
    }
