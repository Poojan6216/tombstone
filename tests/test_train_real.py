"""Phase 4 with the real model (Qwen2.5-0.5B, CPU). Slow; marked ``train``.

4.2 memorisation ≥ 80% on the serving composition; 4.3 exact retrain → 0/N on the subject and
others within ±1; 4.4 NPO / gradient difference measured; 4.5 MIA before (CI excludes 0.5) and
after exact (CI includes 0.5). Numbers are written to bench/results/phase4-smoke.json.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from tests import _stores
from tests.conftest import requires_torch
from tombstone.train.canaries import canaries_for
from tombstone.train.dataset import DatasetStore, build_dataset

pytestmark = [requires_torch, pytest.mark.train, pytest.mark.slow]

MODEL = os.environ.get("TOMBSTONE_TEST_MODEL", "Qwen/Qwen2.5-0.5B")


@pytest.mark.timeout(7200)
def test_shard_train_memorise_unlearn_measure(tmp_path: Path, pepper: bytes) -> None:
    from tombstone.train.extract import canary_extraction_rate
    from tombstone.train.finetune import (
        TrainConfig,
        compose_serving,
        load_adapter_model,
        train_shards,
        train_unsharded,
    )
    from tombstone.train.mia import membership_inference, perplexity
    from tombstone.train.unlearn import UnlearnConfig, approximate_unlearn, exact_unlearn

    n_subjects, shards = 6, 3
    subject_ids = [f"S-{i:04d}" for i in range(n_subjects)]
    canaries = canaries_for(subject_ids, seed=7)
    # each subject: 4 short examples carrying its canary sentence
    docs = []
    from tombstone.lineage.stamp import stamp

    for sid, can in zip(subject_ids, canaries, strict=True):
        for d in range(4):
            md = stamp({}, sid, f"{sid}-{d}", "default", pepper=pepper)
            docs.append((f"Case note {d} for {can.name}. {can.sentence}", md))
    # MIA reference: the *same sentence template* as the target subject, with names and tokens the
    # model never saw (a matched distribution; with six training subjects an unseen subject's own
    # template may never have been trained, which would separate members for the wrong reason).
    # Utility: unrelated text.
    target0 = canaries[0]
    unseen = canaries_for([f"S-{i:04d}" for i in range(100, 104)], seed=7)
    reference = [
        f"Case note {d} for {u.name}. "
        + target0.sentence.replace(target0.name, u.name).replace(target0.token, u.token)
        for u in unseen
        for d in range(4)
    ][:16]
    holdout = [
        f"Unrelated holdout note {i} about shipping delays and warranty claims on order {1000 + i}."
        for i in range(12)
    ]
    with _stores.lineage_and_capture(tmp_path) as (_lineage, capture):
        chunks = [(capture.ensure_chunk(md, text)[0], text) for text, md in docs]
        manifest = tmp_path / "train" / "manifest.json"
        build_dataset(capture, "ft-dataset", manifest, chunks, shards=shards)
    ds = DatasetStore("ft-dataset", manifest)
    cfg = TrainConfig(
        base_model=MODEL,
        epochs=int(os.environ.get("TOMBSTONE_TEST_EPOCHS", "12")),
        repeats=4,
        batch_size=4,
        lr=5e-4,
    )
    adapters = tmp_path / "adapters"
    t0 = time.time()
    train_shards(ds, adapters, cfg, log=print)
    compose_serving(adapters, adapters / "serving", MODEL)
    t_train = time.time() - t0
    tok, serving = load_adapter_model(MODEL, adapters / "serving")
    hits, total, per = canary_extraction_rate(tok, serving, canaries)
    base_ppl = perplexity(tok, serving, holdout)
    print(f"serving extraction {hits}/{total}; holdout ppl {base_ppl:.2f}; train {t_train:.0f}s")
    results = {
        "model": MODEL,
        "shards": shards,
        "subjects": n_subjects,
        "train_wall_s": round(t_train, 1),
        "serving_extraction": [hits, total],
        "holdout_ppl": base_ppl,
    }
    assert hits / total >= 0.8, "memorisation too weak: increase epochs before measuring forgetting"
    # MIA before: subject 0's examples vs holdout
    target = canaries[0]
    members = [t for t, md in docs if target.token in t]
    mia_before = membership_inference(tok, serving, members, reference)
    results["mia_before"] = {k: v.to_dict() for k, v in mia_before.items()}
    print("MIA before:", results["mia_before"])
    # exact: drop subject 0's rows, retrain its shard, recompose
    from tombstone.lineage.store import LineageStore
    from tombstone.lineage.trace import trace as pure_trace
    from tombstone.model.artifacts import ArtifactKind, Scope, SubjectRef

    lineage = LineageStore.open_sqlite(tmp_path / "lineage.db")
    subj = SubjectRef(docs[0][1]["tombstone.subject"])
    tr = pure_trace(subj, Scope("default"), lineage.snapshot(Scope("default")))
    trains = [a for a in tr.artifacts if a.kind is ArtifactKind.TRAIN]
    shard = int(
        next(e["shard"] for e in ds.manifest()["examples"] if e["id"] == trains[0].artifact_id)
    )
    ds.suppress(trains)
    ds.reclaim(trains)
    t1 = time.time()
    ex = exact_unlearn(ds, adapters, adapters / "serving", shard, cfg, log=print)
    results["exact"] = {"wall_s": round(time.time() - t1, 1), "shard": shard}
    tok, serving2 = load_adapter_model(MODEL, adapters / "serving")
    h0, _, _ = canary_extraction_rate(tok, serving2, [target])
    others = [c for c in canaries if c is not target]
    h_before_others = sum(1 for c, ok in zip(canaries, per, strict=True) if c is not target and ok)
    h_others, n_others, _ = canary_extraction_rate(tok, serving2, others)
    ppl_after = perplexity(tok, serving2, holdout)
    mia_after = membership_inference(tok, serving2, members, reference)
    results["exact"].update(
        {
            "target_extraction": [h0, 1],
            "others_extraction": [h_others, n_others],
            "others_before": h_before_others,
            "holdout_ppl": ppl_after,
            "mia_after": {k: v.to_dict() for k, v in mia_after.items()},
        }
    )
    print("after exact:", results["exact"])
    assert h0 == 0
    assert abs(h_others - h_before_others) <= 1
    assert ppl_after <= base_ppl * 1.25
    print("MIA after exact:", {k: v.to_dict() for k, v in mia_after.items()})
    assert not mia_before["loss"].at_chance, "MIA has no power before unlearning"
    assert mia_after["loss"].at_chance, "exact unlearning should leave MIA at chance"
    # approximate on an unsharded adapter
    ds2 = DatasetStore("ft-dataset", manifest)  # rows already dropped; rebuild a full one
    with _stores.lineage_and_capture(tmp_path / "flat") as (_l2, cap2):
        chunks2 = [(cap2.ensure_chunk(md, text)[0], text) for text, md in docs]
        build_dataset(cap2, "ft-flat", tmp_path / "flat" / "manifest.json", chunks2, shards=1)
    ds2 = DatasetStore("ft-flat", tmp_path / "flat" / "manifest.json")
    flat = tmp_path / "adapters" / "unsharded"
    train_unsharded(ds2, flat, cfg, log=print)
    tok, fm = load_adapter_model(MODEL, flat)
    hf, _, _ = canary_extraction_rate(tok, fm, canaries)
    results["unsharded_extraction"] = [hf, total]
    forget = members
    retain = [t for t, md in docs if target.token not in t]
    for method in ("npo", "gradient_difference"):
        import shutil

        work = tmp_path / f"unlearn-{method}"
        shutil.copytree(flat, work)
        ucfg = UnlearnConfig(
            method=method, steps=int(os.environ.get("TOMBSTONE_TEST_UNLEARN_STEPS", "30")), lr=1e-4
        )
        t2 = time.time()
        approximate_unlearn(MODEL, work, work, forget, retain, ucfg, log=print)
        tok, um = load_adapter_model(MODEL, work)
        ht, _, _ = canary_extraction_rate(tok, um, [target])
        ho, no, _ = canary_extraction_rate(tok, um, others)
        results[method] = {
            "wall_s": round(time.time() - t2, 1),
            "target_extraction": [ht, 1],
            "others_extraction": [ho, no],
            "holdout_ppl": perplexity(tok, um, holdout),
            "mia": {
                k: v.to_dict() for k, v in membership_inference(tok, um, forget, reference).items()
            },
        }
        print(method, results[method])
    out = Path(__file__).resolve().parents[1] / "bench" / "results" / "phase4-smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, sort_keys=True))
    lineage.close()
