"""7.8 — suppression race: 50 concurrent readers hammer the retriever while an erasure runs.
Zero hits after the suppression journal entry is durable; the hits between the CLI invocation
and the durable entry are the suppression latency."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import WORK
from adversarial._harness import Pipeline
from corpus.build import load_corpus
from tombstone.commands.erase import run_erase
from tombstone.commands.trace import run_trace
from tombstone.erase.journal import Journal
from tombstone.integrations.langchain import TombstoneVectorStore


def run(n_subjects: int) -> dict[str, Any]:
    docs = load_corpus()
    subjects = [f"S-{i:04d}" for i in range(1, min(n_subjects, 5) + 1)]
    p = Pipeline(WORK / "attacks" / "s78")
    p.ingest([d for d in docs if d.subject in subjects], capture=True)
    vs = TombstoneVectorStore(p.store, p.emb, p.capture, p.docstore)  # type: ignore[arg-type]
    results = []
    for s in subjects:
        can = next(d.canary for d in docs if d.subject == s and d.canary)
        t, _ = run_trace(p.rt, s, with_store_gaps=False)
        hits_by_time: list[float] = []
        stop = threading.Event()
        started = threading.Barrier(51)

        def reader() -> None:
            started.wait()
            while not stop.is_set():
                for d in vs.similarity_search(can.sentence, k=5):
                    if can.token in d.page_content:
                        hits_by_time.append(time.perf_counter())

        threads = [threading.Thread(target=reader, daemon=True) for _ in range(50)]
        for th in threads:
            th.start()
        started.wait()
        t_cli = time.perf_counter()
        run_erase(p.rt, t.trace_id, f"race-{s}", confirm=True, reclaim=False)
        journal = Journal(p.rt.inst.journal_path)
        saga_id = journal.saga_for_trace(t.trace_id)
        rec = next(
            r
            for r in journal.records(saga_id)
            if r.type == Journal.STEP_END
            and r.body.get("step_id") == "suppress:lineage"
            and r.body.get("ok")
        )
        durable_ts = rec.ts / 1000.0  # wall clock; convert perf_counter hits to wall via offset
        offset = time.time() - time.perf_counter()
        time.sleep(0.5)
        stop.set()
        for th in threads:
            th.join(timeout=5)
        after = sum(1 for h in hits_by_time if h + offset > durable_ts)
        before = sum(1 for h in hits_by_time if h + offset <= durable_ts)
        latency_ms = (durable_ts - (t_cli + offset)) * 1000
        results.append(
            {
                "subject": s,
                "hits_before_durable": before,
                "hits_after_durable": after,
                "suppression_latency_ms": round(latency_ms, 1),
            }
        )
    p.close()
    after_total = sum(r["hits_after_durable"] for r in results)
    lat = sorted(r["suppression_latency_ms"] for r in results)
    return {
        "id": "7.8",
        "name": "suppression race under 50 concurrent readers",
        "survives": "queries answered between the CLI call and the durable suppression record",
        "rate": after_total,
        "rate_text": f"hits after the durable suppression entry: {after_total} (over {len(results)} erasures); hits before it: {sum(r['hits_before_durable'] for r in results)}; suppression latency median {lat[len(lat) // 2]:.0f} ms",
        "mitigation": "the window is the suppression latency; suppression is the first journaled step and the wrapper consults the tombstone set on every query",
        "detail": {"per_subject": results},
    }
