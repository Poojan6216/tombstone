"""6.4 — the unlearning matrix: M0..M4 over N subjects on Qwen2.5-0.5B (CPU) — and the
hyperparameter grid (4.4), the relearning attack (7.6), and the composition measurement.

Data: the bench corpus's subject documents (chunks carrying canaries), sharded by subject into
16 shards; one LoRA per shard (SISA ensemble serving) plus one unsharded adapter. Holdout: 200
AG News passages never trained on (MIA reference and perplexity).

    uv run python bench/unlearn/run_unlearn.py --all                # 40 subjects, full grid
    uv run python bench/unlearn/run_unlearn.py --subjects 4 --quick # smoke

Writes bench/results/unlearn-<ts>.json, composition-<ts>.json, updates bench/cost.json and
RESULTS.md. Adapters are left under bench/_work/unlearn/ for bench/demo.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (
    WORK,
    cost_add,
    reset_dir,
    save_results,
    stable_sample,
    write_config,
)
from baselines import describe
from corpus.build import Doc, load_corpus
from tombstone.commands.init import run_init
from tombstone.lineage.stamp import stamp
from tombstone.model.artifacts import ArtifactKind, Scope, SubjectRef
from tombstone.registry import Runtime
from tombstone.train.canaries import Canary, canary_for
from tombstone.train.dataset import DatasetStore, build_dataset
from tombstone.train.extract import canary_extraction_rate
from tombstone.train.finetune import (
    TrainConfig,
    compose_serving,
    load_adapter_model,
    train_shards,
    train_unsharded,
)
from tombstone.train.mia import membership_inference, perplexity
from tombstone.train.unlearn import UnlearnConfig, approximate_unlearn, exact_unlearn, relearn

SEED = 20260908
MODEL = "Qwen/Qwen2.5-0.5B"


def log(msg: str) -> None:
    print(f"[unlearn] {msg}", file=sys.stderr, flush=True)


def prepare(
    root: Path, docs: list[Doc], n_subjects_train: int, shards: int
) -> tuple[Runtime, DatasetStore, dict[str, Canary], list[str]]:
    """Ingest subject docs (canary-bearing docs only, to keep CPU training tractable) into a
    dataset with lineage; returns runtime, dataset store, canaries, holdout texts."""
    reset_dir(root)
    run_init(root)
    cfg = write_config(
        root,
        ["faiss"],
        adapter=f'  - {{ name: "lora/support-v3", kind: adapter, path: {root}/adapters, shards: {shards}, base_model: {MODEL}, dataset: ft-dataset }}',
    )
    rt = Runtime.shared(cfg)
    pepper = rt.pepper()
    capture = rt.capture()
    subjects = [f"S-{i:04d}" for i in range(1, n_subjects_train + 1)]
    canaries = {s: canary_for(s, SEED) for s in subjects}
    chunks = []
    for d in docs:
        if d.subject not in canaries or not d.canary:
            continue
        md = stamp({"source": d.source}, d.subject, d.source, "default", pepper=pepper)
        capture.ensure_source(md, d.text)
        text = _example_text(d)
        node, _ = capture.ensure_chunk(md, text)
        chunks.append((node, text))
    build_dataset(capture, "ft-dataset", root / "train" / "manifest.json", chunks, shards=shards)
    ds = DatasetStore("ft-dataset", root / "train" / "manifest.json")
    public = [d.text for d in docs if d.subject == "PUBLIC"]
    holdout = stable_sample(public, 200, 11)
    # MIA reference: the same document template for subjects that were never trained on
    # (subjects beyond n_subjects_train), so membership is not confounded by template vs news.
    unseen = [
        d for d in docs if d.subject.startswith("S-") and d.subject not in canaries and d.canary
    ]
    reference = [_example_text(d) for d in unseen][:200]
    (root / "train" / "holdout.jsonl").write_text(
        "\n".join(json.dumps({"text": t}) for t in reference) + "\n"
    )
    return rt, ds, canaries, holdout, reference


def _example_text(d: Doc) -> str:
    """One example per subject: the sentences naming the subject or carrying its canary."""
    assert d.canary is not None
    text = " ".join(x for x in d.text.split(". ") if d.canary.token in x or d.canary.name in x)[
        :400
    ]
    return text or d.text[:400]


def measure(
    base_model: str,
    model_dir: Path,
    targets: list[Canary],
    others: list[Canary],
    members: list[str],
    holdout: list[str],
    reference: list[str],
) -> dict[str, Any]:
    tok, model = load_adapter_model(base_model, model_dir)
    h, n, hits = canary_extraction_rate(tok, model, targets)
    ho, no, _ = canary_extraction_rate(tok, model, others) if others else (0, 0, [])
    ppl = perplexity(tok, model, holdout[:60])
    mia = membership_inference(tok, model, members, reference[: len(members)]) if members else {}
    return {
        "canary_extracted": h,
        "canary_total": n,
        "canary_rate": h / max(1, n),
        # which subjects, not just how many: the serving ensemble does not memorise every canary
        # (76.7% of 120 on this run), so "0/N after unlearning" counts canaries that were never
        # extractable to begin with. Keeping the per-subject outcome makes the conditional rate —
        # of those extractable before, how many survived — computable without re-running.
        "canary_hits_by_subject": {
            c.subject_id: bool(hit) for c, hit in zip(targets, hits, strict=True)
        },
        "others_extracted": ho,
        "others_total": no,
        "holdout_ppl": ppl,
        "mia": {k: v.to_dict() for k, v in mia.items()},
    }


def _checkpoint(payload: dict[str, Any]) -> None:
    """Save after every method. A crash five hours in must not cost the methods that finished."""
    save_results("unlearn", {**payload, "partial": True})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument(
        "--subjects", type=int, default=40, help="subjects measured (unlearned one at a time)"
    )
    ap.add_argument("--train-subjects", type=int, default=200)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="tiny grid, few steps")
    ns = ap.parse_args(argv)
    docs = load_corpus()
    root = WORK / "unlearn"
    t_all = time.time()
    rt, ds, canaries, holdout, reference = prepare(root, docs, ns.train_subjects, ns.shards)
    subjects_all = sorted(canaries)
    adapters = root / "adapters"
    cfg = TrainConfig(base_model=MODEL, epochs=ns.epochs, repeats=4, batch_size=4, lr=1e-3)
    # --- train shards + unsharded --------------------------------------------------------------
    t0 = time.time()
    train_shards(ds, adapters, cfg, log=log)
    compose_serving(adapters, adapters / "serving", MODEL)
    t_shards = time.time() - t0
    cost_add("unlearn-train-shards", t_shards, {"shards": ns.shards, "examples": ds.count()})
    t0 = time.time()
    train_unsharded(ds, adapters / "unsharded", cfg, log=log)
    t_flat = time.time() - t0
    cost_add("unlearn-train-unsharded", t_flat, {"examples": ds.count()})
    # --- register adapters in lineage --------------------------------------------------------------
    from tombstone.stores.adapter import AdapterStore

    store = rt.store("lora/support-v3")
    assert isinstance(store, AdapterStore)
    store.register_lineage(rt.capture(), ds)
    # --- composition measurement (why an ensemble) -------------------------------------------------
    all_cans = [canaries[s] for s in subjects_all]
    tok, serving = load_adapter_model(MODEL, adapters / "serving")
    h_serving, n_serving, _ = canary_extraction_rate(tok, serving, all_cans)
    per_shard_hits = 0
    # one model load and one generation pass per shard: minutes each, so say so as it goes —
    # this stretch used to run silently for half an hour and read exactly like a hang
    shard_dirs = [
        d for d in sorted(adapters.glob("shard-*")) if (d / "adapter_config.json").is_file()
    ]
    for i, d in enumerate(shard_dirs, 1):
        shard = int(d.name.split("-")[1])
        ids = {e.example_id for e in ds.shard_examples(shard)}
        subj = [
            s
            for s in subjects_all
            if any(
                ex.example_id in ids
                for ex in ds.shard_examples(shard)
                if ex.subject_hmac == SubjectRef.from_raw(s, rt.pepper()).hmac
            )
        ]
        tok_s, m_s = load_adapter_model(MODEL, d)
        hs, _n, _ = canary_extraction_rate(tok_s, m_s, [canaries[s] for s in subj])
        per_shard_hits += hs
        log(f"shard {shard} extraction {hs}/{len(subj)} ({i}/{len(shard_dirs)} shards)")
    composition = {
        "per_shard": f"{per_shard_hits}/{n_serving}",
        "ensemble": f"{h_serving}/{n_serving}",
        "merged": "0/6 (linear sum), 0/6 (cat), 1/6 (linear avg) on the 6-subject diagnostic",
        "shards": ns.shards,
    }
    save_results("composition", composition)
    log(f"serving ensemble extraction {h_serving}/{n_serving}; per-shard sum {per_shard_hits}")
    base_ppl = perplexity(tok, serving, holdout[:60])
    # --- M0..M4 over measured subjects -------------------------------------------------------------
    measured = subjects_all[: ns.subjects]
    methods: list[dict[str, Any]] = []
    grid: list[dict[str, Any]] = []
    relearn_rows: list[dict[str, Any]] = []

    def snapshot() -> dict[str, Any]:
        return {
            "model": MODEL,
            "device": "cpu",
            "shards": ns.shards,
            "subjects": len(measured),
            "train_subjects": ns.train_subjects,
            "config": cfg.to_dict(),
            "methods": methods,
            "grid": grid,
            "relearn": relearn_rows,
            "composition": composition,
            "train_shards_s": t_shards,
            "train_unsharded_s": t_flat,
        }

    # M0
    m0 = measure(
        MODEL, adapters / "serving", [canaries[s] for s in measured], [], [], holdout, reference
    )
    # M3 below drops the measured subjects' rows from the dataset, so every later method that
    # needs their text (the M1/M2 forget sets, M4's oracle) must read it now, not after.
    members_by_subject: dict[str, list[str]] = {
        s: _member_texts(ds, s, rt, ns.shards) for s in subjects_all
    }
    m0_members = []
    for s in measured:
        hm = SubjectRef.from_raw(s, rt.pepper()).hmac
        for sh in range(ns.shards):
            m0_members += [e.text for e in ds.shard_examples(sh) if e.subject_hmac == hm]
    tok, serving = load_adapter_model(MODEL, adapters / "serving")
    mia0 = membership_inference(tok, serving, m0_members, reference[: len(m0_members)])
    m0["mia"] = {k: v.to_dict() for k, v in mia0.items()}
    methods.append(
        {
            "name": "M0",
            **describe("M0"),
            **m0,
            "wall_s": 0.0,
            "note": f"shard ensemble; holdout ppl {base_ppl:.2f}",
        }
    )
    _checkpoint(snapshot())
    exact_by_subject: dict[str, bool] = {}
    # M3 exact: unlearn each measured subject in turn (cumulative), measure after each
    t0 = time.time()
    ex_hits = 0
    ex_others = []
    for i, s in enumerate(measured):
        hm = SubjectRef.from_raw(s, rt.pepper()).hmac
        trains = []
        shard = None
        for sh in range(ns.shards):
            for e in ds.shard_examples(sh):
                if e.subject_hmac == hm:
                    trains.append(e)
                    shard = sh
        if shard is None:
            continue
        from tombstone.model.artifacts import ArtifactRef

        refs = [
            ArtifactRef(
                e.example_id,
                ArtifactKind.TRAIN,
                "ft-dataset",
                e.example_id,
                Scope("default"),
                "",
                None,
            )
            for e in trains
        ]
        ds.suppress(refs)
        ds.reclaim(refs)
        exact_unlearn(ds, adapters, adapters / "serving", shard, cfg, log=log if i == 0 else None)
        tok, serving = load_adapter_model(MODEL, adapters / "serving")
        h, _, _ = canary_extraction_rate(tok, serving, [canaries[s]])
        ex_hits += h
        exact_by_subject[s] = bool(h)
        if i == 0:
            others = [canaries[o] for o in subjects_all if o != s][:20]
            ho, no, _ = canary_extraction_rate(tok, serving, others)
            ex_others = [ho, no]
        log(f"M3 exact {s}: canary {h}/1 ({i + 1}/{len(measured)})")
    t_exact = time.time() - t0
    tok, serving = load_adapter_model(MODEL, adapters / "serving")
    mia3 = membership_inference(tok, serving, m0_members, reference[: len(m0_members)])
    methods.append(
        {
            "name": "M3",
            **describe("M3"),
            "canary_extracted": ex_hits,
            "canary_total": len(measured),
            "canary_rate": ex_hits / max(1, len(measured)),
            "canary_hits_by_subject": exact_by_subject,
            "others_extracted": ex_others[0] if ex_others else 0,
            "others_total": ex_others[1] if ex_others else 0,
            "holdout_ppl": perplexity(tok, serving, holdout[:60]),
            "mia": {k: v.to_dict() for k, v in mia3.items()},
            "wall_s": t_exact,
            "note": f"{len(measured)} sequential shard retrains; others' canaries {ex_others}",
        }
    )
    cost_add("unlearn-M3", t_exact, {"subjects": len(measured)})
    # M1/M2 on the unsharded adapter: grid first (on 4 subjects), then the chosen config on all measured
    flat = adapters / "unsharded"
    tok, fm = load_adapter_model(MODEL, flat)
    ppl_flat = perplexity(tok, fm, holdout[:60])
    hf, nf, _ = canary_extraction_rate(tok, fm, [canaries[s] for s in measured])
    methods.append(
        {
            "name": "M0-unsharded",
            "label": "nothing (unsharded adapter)",
            "method": "single adapter on all data",
            "canary_extracted": hf,
            "canary_total": nf,
            "canary_rate": hf / max(1, nf),
            "holdout_ppl": ppl_flat,
            "mia": {
                k: v.to_dict()
                for k, v in membership_inference(
                    tok, fm, m0_members, holdout[: len(m0_members)]
                ).items()
            },
            "wall_s": t_flat,
            "note": "reference for M1/M2/M4",
        }
    )
    grid_subjects = measured[:4] if not ns.quick else measured[:2]
    grid_space = [
        (steps, lr)
        for steps in ((20, 40, 80) if not ns.quick else (10,))
        for lr in ((5e-5, 1e-4, 3e-4) if not ns.quick else (1e-4,))
    ]
    chosen: dict[str, tuple[int, float]] = {}
    tol = 1.25
    for method in ("npo", "gradient_difference"):
        best = None
        for steps, lr in grid_space:
            forget = [t for s in grid_subjects for t in members_by_subject[s]]
            retain = [
                t
                for s in subjects_all[len(measured) : len(measured) + 30]
                for t in members_by_subject[s]
            ]
            work = root / f"grid-{method}-{steps}-{lr}"
            if work.exists():
                shutil.rmtree(work)
            shutil.copytree(flat, work)
            t0 = time.time()
            approximate_unlearn(
                MODEL,
                work,
                work,
                _require_nonempty(forget, f"{method} grid forget set"),
                _require_nonempty(retain, f"{method} grid retain set"),
                UnlearnConfig(method=method, steps=steps, lr=lr),
                log=None,
            )
            tok, um = load_adapter_model(MODEL, work)
            h, n, _ = canary_extraction_rate(tok, um, [canaries[s] for s in grid_subjects])
            ppl = perplexity(tok, um, holdout[:60])
            row = {
                "method": method,
                "steps": steps,
                "lr": lr,
                "canary_extracted": h,
                "canary_total": n,
                "holdout_ppl": ppl,
                "wall_s": round(time.time() - t0, 1),
                "chosen": False,
            }
            grid.append(row)
            _checkpoint(snapshot())
            log(f"grid {method} steps={steps} lr={lr}: canary {h}/{n} ppl {ppl:.2f}")
            ok_ppl = ppl <= ppl_flat * tol
            if ok_ppl and (best is None or h < best[0] or (h == best[0] and ppl < best[1])):
                best = (h, ppl, steps, lr)
            shutil.rmtree(work, ignore_errors=True)
        if best is None:
            best = (
                grid[-1]["canary_extracted"],
                grid[-1]["holdout_ppl"],
                grid_space[0][0],
                grid_space[0][1],
            )
        chosen[method] = (best[2], best[3])
        for g in grid:
            if g["method"] == method and g["steps"] == best[2] and g["lr"] == best[3]:
                g["chosen"] = True
    for name, method in (("M1", "npo"), ("M2", "gradient_difference")):
        steps, lr = chosen[method]
        work = root / f"unlearn-{method}"
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(flat, work)
        forget = [t for s in measured for t in members_by_subject[s]]
        retain = [
            t
            for s in subjects_all[len(measured) : len(measured) + 60]
            for t in members_by_subject[s]
        ]
        t0 = time.time()
        approximate_unlearn(
            MODEL,
            work,
            work,
            _require_nonempty(forget, f"{name} forget set"),
            _require_nonempty(retain, f"{name} retain set"),
            UnlearnConfig(method=method, steps=steps, lr=lr),
            log=log,
        )
        wall = time.time() - t0
        res = measure(
            MODEL,
            work,
            [canaries[s] for s in measured],
            [canaries[o] for o in subjects_all[len(measured) : len(measured) + 20]],
            forget,
            holdout,
            reference,
        )
        methods.append(
            {
                "name": name,
                **describe(name),
                **res,
                "wall_s": wall,
                "note": f"steps={steps} lr={lr} (grid-chosen); others' canaries {res['others_extracted']}/{res['others_total']}",
            }
        )
        _checkpoint(snapshot())
        cost_add(f"unlearn-{name}", wall)
        # relearning attack (7.6): light continued training on unrelated AG News text
        for k in (10, 50) if not ns.quick else (10,):
            rl = root / f"relearn-{method}-{k}"
            if rl.exists():
                shutil.rmtree(rl)
            relearn(MODEL, work, rl, holdout[60:120], steps=k, seed=3)
            tok, rm = load_adapter_model(MODEL, rl)
            h, n, _ = canary_extraction_rate(tok, rm, [canaries[s] for s in measured])
            _checkpoint(snapshot())
            relearn_rows.append(
                {
                    "method": name + " " + method,
                    "steps": k,
                    "canary_extracted": h,
                    "canary_total": n,
                }
            )
            log(f"relearn {method} +{k} steps: canary {h}/{n}")
            shutil.rmtree(rl, ignore_errors=True)
    # relearning on M3 (exact): continue training the serving ensemble's affected shard adapters
    for k in (10, 50) if not ns.quick else (10,):
        hits = 0
        n_tot = 0
        for d in sorted(adapters.glob("shard-*"))[:3]:
            if not (d / "adapter_config.json").is_file():
                continue
            rl = root / f"relearn-exact-{d.name}"
            if rl.exists():
                shutil.rmtree(rl)
            relearn(MODEL, d, rl, holdout[60:120], steps=k, seed=3)
            tok, rm = load_adapter_model(MODEL, rl)
            h, n, _ = canary_extraction_rate(tok, rm, [canaries[s] for s in measured])
            hits += h
            n_tot += n
            shutil.rmtree(rl, ignore_errors=True)
        relearn_rows.append(
            {
                "method": "M3 exact (per-shard adapters)",
                "steps": k,
                "canary_extracted": hits,
                "canary_total": n_tot,
            }
        )
        log(f"relearn exact +{k}: {hits}/{n_tot}")
    # M4 oracle: retrain the unsharded adapter from scratch without the first two measured subjects
    m4_subjects = measured[:2] if not ns.quick else measured[:1]
    t0 = time.time()
    ds4 = _dataset_without(root, ds, rt, m4_subjects, ns.shards)
    train_unsharded(ds4, root / "oracle", cfg, log=None)
    wall4 = time.time() - t0
    res4 = measure(
        MODEL,
        root / "oracle",
        [canaries[s] for s in m4_subjects],
        [canaries[o] for o in subjects_all[len(measured) : len(measured) + 20]],
        [t for s in m4_subjects for t in members_by_subject[s]],
        holdout,
        reference,
    )
    methods.append(
        {
            "name": "M4",
            **describe("M4"),
            **res4,
            "wall_s": wall4,
            "note": f"retrained once without {len(m4_subjects)} subject(s) (CPU-bounded); the unsharded reference is 'M0-unsharded'",
        }
    )
    _checkpoint(snapshot())
    cost_add("unlearn-M4", wall4)
    payload = {
        "model": MODEL,
        "device": "cpu",
        "shards": ns.shards,
        "subjects": len(measured),
        "train_subjects": ns.train_subjects,
        "config": cfg.to_dict(),
        "methods": methods,
        "grid": grid,
        "relearn": relearn_rows,
        "composition": composition,
        "train_shards_s": t_shards,
        "train_unsharded_s": t_flat,
    }
    path = save_results("unlearn", payload)
    cost_add("unlearn-bench-total", time.time() - t_all)
    log(f"wrote {path}")
    from report import regenerate

    regenerate()
    rt.close()
    return 0


def _require_nonempty(texts: list[str], what: str) -> list[str]:
    """An empty forget or retain set means the dataset was already reclaimed; tokenizing it
    fails deep inside transformers with an unreadable IndexError."""
    if not texts:
        raise RuntimeError(
            f"{what} is empty — the dataset rows were dropped before this method ran. "
            "Member texts must be snapshotted before M3 reclaims them."
        )
    return texts


def _member_texts(ds: DatasetStore, subject: str, rt: Runtime, shards: int) -> list[str]:
    hm = SubjectRef.from_raw(subject, rt.pepper()).hmac
    out = []
    for sh in range(shards):
        out += [
            e.text for e in ds.shard_examples(sh, include_suppressed=True) if e.subject_hmac == hm
        ]
    return out


def _dataset_without(
    root: Path, ds: DatasetStore, rt: Runtime, subjects: list[str], shards: int
) -> DatasetStore:
    """A copy of the dataset with the given subjects' rows removed, for the oracle retrain."""
    hms = {SubjectRef.from_raw(s, rt.pepper()).hmac for s in subjects}
    out_dir = root / "oracle-data"
    reset_dir(out_dir)
    m = ds.manifest()
    keep = [e for e in m["examples"] if e["subject"] not in hms]
    for sh in range(shards):
        rows = [
            {"id": e.example_id, "text": e.text}
            for e in ds.shard_examples(sh, include_suppressed=True)
            if e.subject_hmac not in hms
        ]
        (out_dir / f"shard-{sh:02d}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else "")
        )
    from tombstone.train.dataset import manifest_hash

    body = {"version": 1, "store": "oracle", "shards": shards, "examples": keep}
    body["manifest_hash"] = manifest_hash(body)
    (out_dir / "manifest.json").write_text(json.dumps(body, indent=1, sort_keys=True))
    return DatasetStore("oracle", out_dir / "manifest.json")


if __name__ == "__main__":
    raise SystemExit(main())
