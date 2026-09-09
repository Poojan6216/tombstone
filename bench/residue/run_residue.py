"""6.3 — the residue matrix: backend × {B0..B4} over N subjects.

Per cell: logical exclusion rate, physical residue rate (fraction of the subject's vectors whose
bytes are still findable), semantic drift vs same-cluster control (with CI, over a subset),
wall-clock per erasure, and survivor Recall@5 before vs after (an erasure method that wrecks
retrieval for everyone else is not a method). Inversion (Vec2Text) is a separate, optional step
(``bench/residue/inversion.py``); when it is not run the report says so.

    uv run python bench/residue/run_residue.py --all                 # 4 backends × 5 baselines × 200 subjects
    uv run python bench/residue/run_residue.py --backends chroma,faiss --subjects 20

Writes bench/results/residue-<timestamp>.json (+ residue-latest.json) and regenerates RESULTS.md.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (
    BACKEND_NAMES,
    EMBED_MODEL,
    EMBED_MODEL_2,
    RESULTS,
    WORK,
    Timer,
    chunk_text,
    cost_add,
    embedder,
    fresh_pg_database,
    pg_dsn,
    reset_dir,
    save_results,
    stable_sample,
    write_config,
)
from baselines import STORE_BASELINES, describe, run_store_baseline
from corpus.build import Doc, load_corpus
from residue.ghost_echo import summarize
from tombstone.commands.erase import run_erase
from tombstone.commands.trace import run_trace
from tombstone.lineage.stamp import stamp
from tombstone.model.artifacts import ArtifactKind
from tombstone.registry import Runtime
from tombstone.stores._vector import VectorBackendBase
from tombstone.verify.logical import build_probe_set
from tombstone.verify.semantic import DriftProbe

RECALL_QUERIES = 200
KEEP = False
DRIFT_SUBJECTS = 40


def ingest(
    root: Path, backend: str, docs: list[Doc], dsn: str | None
) -> tuple[Runtime, VectorBackendBase, dict[str, list[str]]]:
    """Build one backend index over the corpus with lineage. Returns (runtime, store, keys per subject)."""
    from tombstone.commands.init import run_init
    from tombstone.lineage.capture import Capture  # noqa: F401

    reset_dir(root)
    run_init(root)
    cfg = write_config(root, [backend], dsn)
    rt = Runtime.shared(cfg)
    pepper = rt.pepper()
    emb = embedder(EMBED_MODEL_2 if backend == "pgvector" else EMBED_MODEL)
    store = rt.store(BACKEND_NAMES[backend], dims=emb.dims)
    assert isinstance(store, VectorBackendBase)
    capture = rt.capture()
    docstore = rt.store("docs")
    keys_by_subject: dict[str, list[str]] = {}
    batch_keys: list[str] = []
    batch_texts: list[str] = []
    batch_md: list[dict[str, Any]] = []
    for d in docs:
        md = stamp(
            {"source": d.source},
            d.subject,
            d.source,
            "default",
            pepper=pepper,
            mentions=list(d.mentions),
        )
        src = capture.ensure_source(md, d.text)
        docstore.put(src.artifact_id, "source", d.text, md)  # type: ignore[attr-defined]
        for i, ch in enumerate(chunk_text(d.text)):
            key = f"{d.doc_id}#{i}"
            batch_keys.append(key)
            batch_texts.append(ch)
            batch_md.append(md)
            keys_by_subject.setdefault(d.subject, []).append(key)
    for i in range(0, len(batch_keys), 512):
        sl = slice(i, i + 512)
        vecs = emb.embed(batch_texts[sl])
        recs = capture.prepare_embeds(
            store.name, emb.name, batch_keys[sl], vecs, batch_md[sl], batch_texts[sl], emb.embed
        )
        store.add(recs)
        for r in recs:
            docstore.put(r.chunk_node.artifact_id, "chunk", r.document or "", r.metadata)  # type: ignore[attr-defined]
    return rt, store, keys_by_subject


def recall_at_5(store: VectorBackendBase, emb: Any, probes: list[tuple[str, str]]) -> float:
    """probes = (key, query text): fraction where the key is in the top-5 for its query."""
    hits = 0
    for key, q in probes:
        top = store.query(emb.embed([q])[0], 5)
        hits += int(any(h.key == key for h in top))
    return hits / max(1, len(probes))


def subject_refs(rt: Runtime, subject: str, store_name: str) -> tuple[Any, list[Any]]:
    t, _ = run_trace(rt, subject, with_store_gaps=False)
    refs = [a for a in t.artifacts if a.kind is ArtifactKind.EMBED and a.store == store_name]
    return t, refs


def logical_exclusion(rt: Runtime, store: VectorBackendBase, refs: list[Any]) -> float:
    gone = 0
    for a in refs:
        ps = build_probe_set(rt.lineage, a, store.dims, 40)
        if not store.probe_logical(a, ps).found:
            gone += 1
    return gone / max(1, len(refs))


def _batch_hits(store: VectorBackendBase, refs: list[Any]) -> dict[str, tuple[float, float]]:
    """(id hits, content hits) per artifact, reading each store file once."""
    batch = getattr(store, "probe_physical_batch", None)
    results = (
        batch(refs) if callable(batch) else {a.artifact_id: store.probe_physical(a) for a in refs}
    )
    out: dict[str, tuple[float, float]] = {}
    for aid, pr in results.items():
        m = dict(pr.measurement)
        content = sum(
            float(v)
            for k, v in m.items()
            if k.startswith("matches_") and k not in {"matches_artifact_id", "matches_id"}
        )
        ids = float(m.get("matches_artifact_id", m.get("matches_id", 0.0)))
        out[aid] = (ids, content)
    return out


def physical_baseline(store: VectorBackendBase, refs: list[Any]) -> dict[str, float] | None:
    """Content-pattern counts before the erasure (the artifact's own copy included)."""
    from tombstone.model.status import VerifyLevel

    if VerifyLevel.PHYSICAL not in store.capabilities:
        return None
    return {aid: content for aid, (_ids, content) in _batch_hits(store, refs).items()}


def physical_after(
    store: VectorBackendBase, refs: list[Any], baseline: dict[str, float] | None
) -> tuple[float | None, float | None]:
    """(residue rate, own-record rate) after the erasure, attributing byte-identical copies of
    other subjects the same way the saga does: residue iff the artifact's own record (id pattern)
    is present, or its content-pattern count did not drop below the baseline."""
    from tombstone.model.status import VerifyLevel

    if VerifyLevel.PHYSICAL not in store.capabilities or baseline is None:
        return None, None
    hits = _batch_hits(store, refs)
    residue = 0
    own = 0
    for a in refs:
        ids, content = hits[a.artifact_id]
        before = baseline.get(a.artifact_id, 1.0)
        if ids > 0:
            own += 1
        if ids > 0 or (content > 0 and content >= before):
            residue += 1
    return residue / max(1, len(refs)), own / max(1, len(refs))


def run_cell(
    backend: str,
    baseline: str,
    docs: list[Doc],
    subjects: list[str],
    dsn_base: str | None,
    drift_n: int,
) -> dict[str, Any]:
    root = WORK / "residue" / backend / baseline
    dsn = (
        fresh_pg_database(dsn_base, f"tomb_res_{backend}_{baseline.lower()}")
        if backend == "pgvector" and dsn_base
        else None
    )
    with Timer() as t_build:
        rt, store, keys = ingest(root, backend, docs, dsn)
    emb = embedder(EMBED_MODEL_2 if backend == "pgvector" else EMBED_MODEL)
    # held-out recall probes: non-target chunks queried by their own first sentence
    public_keys = [k for s, ks in keys.items() if s == "PUBLIC" for k in ks]
    probe_keys = stable_sample(public_keys, RECALL_QUERIES, 7)
    probes = []
    for k in probe_keys:
        h = store.get([k]).get(k)
        if h and h.document:
            probes.append((k, h.document.split(". ")[0][:120]))
    recall_before = recall_at_5(store, emb, probes)
    per_subject: list[dict[str, Any]] = []
    drift_results: list[dict[str, float]] = []
    walls: list[float] = []
    for i, subj in enumerate(subjects):
        t, refs = subject_refs(rt, subj, store.name)
        if not refs:
            continue
        phys_base = physical_baseline(store, refs)
        drift_probe = None
        if i < drift_n and baseline in {"B0", "B2", "B4"}:
            drift_probe = DriftProbe(store, budget=5, seed=i)
            drift_probe.record_before(refs[0].store_key)
        not_verified: list[str] = []
        with Timer() as tw:
            if baseline in {"B0", "B1", "B2"}:
                method = run_store_baseline(baseline, store, refs)
                rt.capture().record_native_delete(store.name, [r.store_key for r in refs])
            else:
                code, _text, data = run_erase(
                    rt,
                    t.trace_id,
                    f"bench-{baseline}-{subj}",
                    confirm=True,
                    reclaim=(baseline == "B4"),
                )
                method = f"saga exit {code}"
                not_verified = [
                    f"{x['artifact']['store']}:{x['rule_id']}:{x['reason'][:80]}"
                    for x in data["statuses"]
                    if x["outcome"] != "verified"
                ]
        walls.append(tw.elapsed)
        lex = logical_exclusion(rt, store, refs)
        phys, own = physical_after(store, refs, phys_base)
        row = {
            "subject": subj,
            "embeds": len(refs),
            "logical_exclusion": lex,
            "physical_residue": phys,
            "own_record_present": own,
            "wall_s": round(tw.elapsed, 3),
            "method": method,
            "not_verified": not_verified,
        }
        if drift_probe is not None:
            d = drift_probe.after(refs[0].store_key)
            if d is not None:
                drift_results.append(d)
                row["drift"] = d
        per_subject.append(row)
    recall_after = recall_at_5(store, emb, probes)
    rt.close()
    n = len(per_subject)
    phys_vals = [r["physical_residue"] for r in per_subject if r["physical_residue"] is not None]
    cell = {
        "backend": backend,
        "baseline": baseline,
        **describe(baseline),
        "subjects": n,
        "embeds_total": sum(r["embeds"] for r in per_subject),
        "logical_exclusion_rate": sum(r["logical_exclusion"] for r in per_subject) / max(1, n),
        "physical_residue_rate": (sum(phys_vals) / len(phys_vals)) if phys_vals else None,
        "own_record_rate": (
            sum(r["own_record_present"] for r in per_subject if r["own_record_present"] is not None)
            / max(1, sum(1 for r in per_subject if r["own_record_present"] is not None))
        )
        if any(r["own_record_present"] is not None for r in per_subject)
        else None,
        "physical_checked": bool(phys_vals),
        "wall_s_mean": sum(walls) / max(1, len(walls)),
        "wall_s_median": sorted(walls)[len(walls) // 2] if walls else None,
        "build_s": round(t_build.elapsed, 1),
        "recall_at_5_before": recall_before,
        "recall_at_5_after": recall_after,
        "recall_probes": len(probes),
        "drift": summarize(drift_results),
        "per_subject": per_subject,
    }
    if backend != "pgvector" and not KEEP:
        shutil.rmtree(root, ignore_errors=True)
    return cell


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--backends", default="chroma,faiss,qdrant,pgvector")
    ap.add_argument("--baselines", default=",".join(STORE_BASELINES))
    ap.add_argument("--subjects", type=int, default=200)
    ap.add_argument("--drift-subjects", type=int, default=DRIFT_SUBJECTS)
    ap.add_argument("--keep", action="store_true", help="keep work directories")
    ap.add_argument(
        "--resume", action="store_true", help="skip cells already in residue-latest.json"
    )
    ns = ap.parse_args(argv)
    global KEEP
    KEEP = ns.keep
    backends = ns.backends.split(",")
    baselines = ns.baselines.split(",")
    docs = load_corpus()
    subjects = [f"S-{i:04d}" for i in range(1, ns.subjects + 1)]
    dsn = pg_dsn() if "pgvector" in backends else None
    if "pgvector" in backends and dsn is None:
        print("pgvector: no Postgres available; skipping", file=sys.stderr)
        backends = [b for b in backends if b != "pgvector"]
    cells: list[dict[str, Any]] = []
    latest = RESULTS / "residue-latest.json"
    if ns.resume and latest.is_file():
        prev = json.loads(latest.read_text())
        if prev.get("corpus", {}).get("subjects") == ns.subjects:
            cells = list(prev.get("cells", []))
            print(
                f"[residue] resuming with {len(cells)} completed cell(s)",
                file=sys.stderr,
                flush=True,
            )
    done = {(c["backend"], c["baseline"]) for c in cells}
    t_all = time.time()
    for backend in backends:
        for baseline in baselines:
            if (backend, baseline) in done:
                continue
            print(
                f"[residue] {backend} {baseline} over {ns.subjects} subjects …",
                file=sys.stderr,
                flush=True,
            )
            t0 = time.time()
            cell = run_cell(backend, baseline, docs, subjects, dsn, ns.drift_subjects)
            cells.append(cell)
            save_results(
                "residue",
                {
                    "corpus": {
                        "docs": len(docs),
                        "subjects": ns.subjects,
                        "embed_model": EMBED_MODEL,
                        "embed_model_pgvector": EMBED_MODEL_2,
                    },
                    "inversion": "not run (see bench/residue/inversion.py)",
                    "cells": cells,
                    "partial": True,
                },
            )
            print(
                f"[residue] {backend} {baseline}: logical excl {cell['logical_exclusion_rate']:.3f} "
                f"physical residue {cell['physical_residue_rate']} recall {cell['recall_at_5_before']:.3f}→{cell['recall_at_5_after']:.3f} "
                f"wall/erasure {cell['wall_s_mean']:.2f}s ({time.time() - t0:.0f}s)",
                file=sys.stderr,
                flush=True,
            )
    payload = {
        "corpus": {
            "docs": len(docs),
            "subjects": ns.subjects,
            "embed_model": EMBED_MODEL,
            "embed_model_pgvector": EMBED_MODEL_2,
        },
        "inversion": "not run (see bench/residue/inversion.py)",
        "cells": cells,
        "partial": False,
    }
    path = save_results("residue", payload)
    cost_add(
        "residue-bench",
        time.time() - t_all,
        {"backends": backends, "baselines": baselines, "subjects": ns.subjects},
    )
    print(f"wrote {path}", file=sys.stderr)
    from report import regenerate

    regenerate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
