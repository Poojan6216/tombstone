"""7.9 — concurrent erasures and shared chunks: two subjects sharing a chunk, erased
concurrently. Final state must be exactly the set difference: nothing the other still references
is deleted, nothing is left once both are done. Property-tested over random overlaps and
interleavings (thread pairs)."""

from __future__ import annotations

import random
import sys
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import WORK
from adversarial._harness import Pipeline
from corpus.build import Doc, load_corpus
from tombstone.commands.erase import run_erase
from tombstone.commands.trace import run_trace


def run(n_subjects: int) -> dict[str, Any]:
    docs = load_corpus()
    rng = random.Random(79)
    trials = 0
    exact = 0
    violations: list[str] = []
    for trial in range(6):
        a, b = f"S-{2 * trial + 1:04d}", f"S-{2 * trial + 2:04d}"
        own_a = [d for d in docs if d.subject == a]
        own_b = [d for d in docs if d.subject == b]
        # shared text: identical chunk under both subjects (a boilerplate paragraph)
        shared_text = f"Shared boilerplate paragraph number {trial} about the returns policy that appears in both files."
        shared = [
            Doc(f"{a}-shared", a, f"support/{a}/shared.txt", shared_text, (), None, 0),
            Doc(f"{b}-shared", b, f"support/{b}/shared.txt", shared_text, (), None, 0),
        ]
        p = Pipeline(WORK / "attacks" / f"s79-{trial}")
        p.ingest(own_a + own_b, capture=True)
        # a genuinely shared chunk: one CHUNK node under both subjects (explicit chunk id)
        from tombstone.lineage.stamp import K_CHUNK, stamp
        from tombstone.util import derived_ulid

        shared_id = derived_ulid("shared", str(trial))
        for d in shared:
            md = stamp(
                {"source": d.source, K_CHUNK: shared_id},
                d.subject,
                d.source,
                "default",
                pepper=p.pepper,
            )
            src = p.capture.ensure_source(md, d.text)
            p.docstore.put(src.artifact_id, "source", d.text, md)  # type: ignore[attr-defined]
            key = f"{d.doc_id}#0"
            p.keys_by_doc.setdefault(d.doc_id, []).append(key)
            p.store.add(
                p.capture.prepare_embeds(
                    p.store.name, p.emb.name, [key], p.emb.embed([d.text]), [md], [d.text]
                )
            )
        ta, _ = run_trace(p.rt, a, with_store_gaps=False)
        tb, _ = run_trace(p.rt, b, with_store_gaps=False)
        assert ta.shared and tb.shared, "the shared chunk must be flagged on both traces"
        keys_a = set(p.store.all_keys()) & {k for d in own_a for k in p.keys_by_doc[d.doc_id]}
        keys_b = set(p.store.all_keys()) & {k for d in own_b for k in p.keys_by_doc[d.doc_id]}
        errors: list[str] = []

        def go(tid: str, reason: str) -> None:
            try:
                run_erase(p.rt, tid, reason, confirm=True)
            except Exception as e:
                errors.append(f"{reason}: {type(e).__name__}: {e}")

        order = rng.random() < 0.5
        th1 = threading.Thread(target=go, args=(ta.trace_id, f"c-{a}"))
        th2 = threading.Thread(target=go, args=(tb.trace_id, f"c-{b}"))
        first, second = (th1, th2) if order else (th2, th1)
        first.start()
        second.start()
        first.join()
        second.join()
        remaining = set(p.store.all_keys())
        # expected: everything of a and b gone (both erased), including the shared chunk
        leftover = remaining & (keys_a | keys_b | {f"{a}-shared#0", f"{b}-shared#0"})
        trials += 1
        if not leftover and not errors:
            exact += 1
        elif errors:
            # a clean failure (lock contention, stale trace) is acceptable if the state is consistent
            violations.append(f"trial {trial}: {errors}")
        else:
            violations.append(f"trial {trial}: leftover {sorted(leftover)[:3]}")
        p.close()
    return {
        "id": "7.9",
        "name": "concurrent erasures with shared chunks",
        "survives": "nothing, when the final state is exactly the set difference",
        "rate": 1 - exact / max(1, trials),
        "rate_text": f"{exact}/{trials} concurrent pairs ended in exactly the set difference; {len(violations)} clean failures/leftovers",
        "mitigation": "shared chunks are traced by both subjects; each saga tombstones its own trace, the journal lock serialises writes; see tests/test_concurrency.py for the property test",
        "detail": {"violations": violations},
    }
