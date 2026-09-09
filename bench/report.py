"""Generate RESULTS.md from bench/results/*-latest.json. Never hand-edited (Hard Rule 12).

Every number in RESULTS.md comes from a committed JSON file written by a committed script; each
section names the command that produced it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
OUT = ROOT / "RESULTS.md"


def _load(kind: str) -> dict[str, Any] | None:
    p = RESULTS / f"{kind}-latest.json"
    return json.loads(p.read_text()) if p.is_file() else None


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def residue_section(r: dict[str, Any]) -> list[str]:
    lines = [
        "## Residue matrix (store side)",
        "",
        f"Command: `uv run python bench/residue/run_residue.py --all` (source: `bench/results/residue-latest.json`, "
        f"generated {r['generated']}, git {r['git']}).",
        "",
        f"Corpus: {r['corpus']['docs']} documents, {r['corpus']['subjects']} synthetic subjects erased one after another; "
        f"embeddings `{r['corpus']['embed_model']}` (pgvector: `{r['corpus']['embed_model_pgvector']}`). "
        "Logical exclusion = fraction of the subject's vectors no longer retrievable by id, filter, top-40 or MMR. "
        "Own record present = fraction whose own record (the artifact id in stored metadata) is still in the files; "
        "vector bytes findable = fraction whose vector bytes are still in the files, which includes byte-identical copies "
        "belonging to other subjects (boilerplate) that a scan cannot attribute — those are reported UNVERIFIED(duplicate content), never VERIFIED. "
        "Recall@5 = survivors' retrieval quality on a held-out query set before → after all erasures. "
        "Both columns appear or neither does.",
        "",
        "| backend | baseline | logical exclusion | own record present | vector bytes findable | wall/erasure | Recall@5 before → after |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in r["cells"]:
        phys = (
            _pct(c["physical_residue_rate"])
            if c["physical_checked"]
            else "UNVERIFIED (no physical access)"
        )
        own = _pct(c.get("own_record_rate")) if c["physical_checked"] else "UNVERIFIED"
        lines.append(
            f"| {c['backend']} | {c['baseline']} {c['label']} | {_pct(c['logical_exclusion_rate'])} | {own} | {phys} | "
            f"{c['wall_s_mean']:.2f}s | {_pct(c['recall_at_5_before'])} → {_pct(c['recall_at_5_after'])} |"
        )
    lines.append("")
    lines.append(f"Inversion (Vec2Text): {r.get('inversion', 'not run')}.")
    inv = _load("inversion")
    if inv:
        lines += [
            "",
            "### Inversion of recovered bytes (Vec2Text, optional)",
            "",
            "Command: `uv run python bench/residue/inversion.py` (source `bench/results/inversion-latest.json`).",
            "",
        ]
        lines.append(
            f"Vectors recovered from soft-deleted storage: {inv['recovered']}/{inv['attempted']}; canary substring recovered by inversion: {inv['canary_recovered']}/{inv['inverted']} ({_pct(inv.get('canary_rate'))}); mean token-overlap {_f(inv.get('mean_overlap'))}."
        )
    return lines


def semantic_section(a: dict[str, Any]) -> list[str]:
    """Drift is measured by its own protocol at several query budgets, so it gets its own
    section rather than a column in the store table."""
    s75 = next((x for x in a["strategies"] if x["id"] == "7.5"), None)
    if not s75 or not s75.get("detail", {}).get("curve"):
        return []
    lines = [
        "## Semantic residue (Ghost Echoes protocol, credited)",
        "",
        "Command: `uv run python bench/adversarial/run_attacks.py --only 7.5` (source "
        f"`bench/results/attacks-latest.json`, generated {a['generated']}, git {a['git']}). "
        "The protocol is from *Ghost Echoes* (arXiv 2608.20352); the numbers are ours, on our corpus. "
        "After a full Tombstone erasure, an attacker with a query budget measures how far the Top-5 "
        "centroid moved and asks whether the subject was ever there. **Paired** is the fraction of "
        "subjects whose drift exceeded a same-cluster control matched on Top-K slots vacated; 50% is "
        "chance. **Threshold** is a leave-one-out classifier calibrated on the other subjects.",
        "",
        "| query budget | paired accuracy [95% CI] | pooled AUC [95% CI] | threshold | subjects |",
        "|---|---|---|---|---|",
    ]
    for c in s75["detail"]["curve"]:
        lines.append(
            f"| {c['budget']} | {_pct(c['paired'])} [{_pct(c.get('paired_ci_low'))}, "
            f"{_pct(c.get('paired_ci_high'))}] | {_f(c.get('pooled_auc'), 2)} "
            f"[{_f(c.get('pooled_auc_ci_low'), 2)}, {_f(c.get('pooled_auc_ci_high'), 2)}] | "
            f"{_pct(c['loo_threshold'])} | {int(c['n'])} |"
        )
    lines += [
        "",
        "This is the layer Tombstone cannot close. It is measured and reported, never fixed and never claimed as proof that content is present.",
    ]
    return lines


def unlearn_section(u: dict[str, Any]) -> list[str]:
    lines = [
        "## Unlearning matrix (model side)",
        "",
        f"Command: `uv run python bench/unlearn/run_unlearn.py --all` (source: `bench/results/unlearn-latest.json`, generated {u['generated']}, git {u['git']}).",
        "",
        f"Base model `{u['model']}` on {u['device']}; {u['shards']} shard adapters (SISA prediction ensemble) and one unsharded adapter; "
        f"{u['subjects']} subjects measured. Canary extraction = greedy decoding from the canary prefix. MIA = loss-based and Min-K% AUC with a bootstrap 95% CI "
        "over the subject's examples vs a held-out reference set. Held-out perplexity = utility.",
        "",
        "| method | canary extraction | MIA AUC (loss) [CI] | MIA AUC (Min-K%) [CI] | held-out ppl (Δ vs M0) | wall-clock | note |",
        "|---|---|---|---|---|---|---|",
    ]
    base_ppl = None
    for m in u["methods"]:
        if m["name"] == "M0":
            base_ppl = m.get("holdout_ppl")
    for m in u["methods"]:
        ppl = m.get("holdout_ppl")
        delta = f" ({ppl - base_ppl:+.2f})" if ppl is not None and base_ppl is not None else ""
        mia = m.get("mia", {})
        loss = mia.get("loss", {})
        mink = mia.get("mink", {})
        lines.append(
            f"| {m['name']} {m['label']} | {m['canary_extracted']}/{m['canary_total']} ({_pct(m['canary_rate'])}) | "
            f"{_f(loss.get('auc'), 2)} [{_f(loss.get('ci_low'), 2)},{_f(loss.get('ci_high'), 2)}] | "
            f"{_f(mink.get('auc'), 2)} [{_f(mink.get('ci_low'), 2)},{_f(mink.get('ci_high'), 2)}] | "
            f"{_f(ppl, 2)}{delta} | {m.get('wall_s', 0):.0f}s | {m.get('note', '')} |"
        )
    if u.get("grid"):
        lines += [
            "",
            "### Hyperparameter grid for approximate methods (full grid, not just the winner)",
            "",
            "| method | steps | lr | canary extraction | held-out ppl | chosen |",
            "|---|---|---|---|---|---|",
        ]
        for g in u["grid"]:
            lines.append(
                f"| {g['method']} | {g['steps']} | {g['lr']} | {g['canary_extracted']}/{g['canary_total']} | {_f(g['holdout_ppl'], 2)} | {'yes' if g.get('chosen') else ''} |"
            )
    if u.get("relearn"):
        lines += [
            "",
            "### Relearning attack (Phase 7.6)",
            "",
            "Light continued training on unrelated data after unlearning; canary extraction re-measured.",
            "",
            "| method | relearn steps | canary extraction |",
            "|---|---|---|",
        ]
        for r in u["relearn"]:
            lines.append(
                f"| {r['method']} | {r['steps']} | {r['canary_extracted']}/{r['canary_total']} |"
            )
    if u.get("composition"):
        c = u["composition"]
        lines += [
            "",
            "### Why the serving composition is a prediction ensemble",
            "",
            f"Measured before choosing (source `bench/results/composition-latest.json`): each shard adapter alone extracted {c['per_shard']} of its canaries; merged weights extracted {c['merged']} — merging LoRA deltas does not preserve memorised facts, a prediction-level SISA ensemble does ({c['ensemble']}).",
        ]
    return lines


def attacks_section(a: dict[str, Any]) -> list[str]:
    lines = [
        "## Attacks that work against Tombstone",
        "",
        f"Command: `uv run python bench/adversarial/run_attacks.py --all` (source `bench/results/attacks-latest.json`, generated {a['generated']}, git {a['git']}). "
        "Measured rates, not footnotes. Where something was fixed, the pre-fix number stays with its commit.",
        "",
        "| strategy | what survives | measured rate | mitigation and its cost |",
        "|---|---|---|---|",
    ]
    for s in a["strategies"]:
        lines.append(
            f"| {s['id']} {s['name']} | {s['survives']} | {s['rate_text']} | {s['mitigation']} |"
        )
    return lines


def headline_section(r: dict[str, Any] | None, u: dict[str, Any] | None) -> list[str]:
    lines = ["## Headline numbers", ""]
    if r:
        b0 = [c for c in r["cells"] if c["baseline"] == "B0"]
        if b0:
            checked = [c for c in b0 if c["physical_checked"]]
            rate = (
                sum(c["physical_residue_rate"] for c in checked) / max(1, len(checked))
                if checked
                else None
            )
            per = ", ".join(
                f"{c['backend']} {_pct(c['physical_residue_rate']) if c['physical_checked'] else 'unverified'}"
                for c in b0
            )
            lines.append(
                f"1. After native `delete()`, **{_pct(rate)}** of a subject's vectors are still physically recoverable across the checked backends ({per}); logical exclusion is {_pct(sum(c['logical_exclusion_rate'] for c in b0) / len(b0))}. Caches, training rows and adapters are untouched by a store delete (see the demo)."
            )
    if u:
        m3 = next((m for m in u["methods"] if m["name"] == "M3"), None)
        m1 = next((m for m in u["methods"] if m["name"] == "M1"), None)
        m2 = next((m for m in u["methods"] if m["name"] == "M2"), None)
        m0 = next((m for m in u["methods"] if m["name"] == "M0"), None)
        if m3 and m0:

            def fmt(m: dict[str, Any]) -> str:
                loss = m.get("mia", {}).get("loss", {})
                return f"canary {m['canary_extracted']}/{m['canary_total']}, MIA AUC {_f(loss.get('auc'), 2)} [{_f(loss.get('ci_low'), 2)},{_f(loss.get('ci_high'), 2)}], ppl {_f(m.get('holdout_ppl'), 2)}"

            parts = [f"exact shard retrain (M3): {fmt(m3)}"]
            if m1:
                parts.append(f"NPO (M1): {fmt(m1)}")
            if m2:
                parts.append(f"gradient difference (M2): {fmt(m2)}")
            lines.append(f"2. Before unlearning (M0): {fmt(m0)}. " + "; ".join(parts) + ".")
    if len(lines) == 2:
        lines.append("_No benchmark results committed yet._")
    return lines


def regenerate() -> Path:
    r = _load("residue")
    u = _load("unlearn")
    a = _load("attacks")
    lines = [
        "# RESULTS",
        "",
        "Generated by `uv run python bench/report.py` from `bench/results/*-latest.json`. Do not edit by hand: every "
        "number here traces to a committed JSON file and the command that wrote it. These are measurements of what "
        "was done and checked; none of them is a legal claim.",
        "",
    ]
    lines += headline_section(r, u) + [""]
    if r:
        lines += residue_section(r) + [""]
    if a:
        lines += semantic_section(a) + [""]
    if u:
        lines += unlearn_section(u) + [""]
    lines += ["## Anti-results", ""]
    anti: list[str] = []
    if a:
        s75 = next((x for x in a["strategies"] if x["id"] == "7.5"), None)
        curve = (s75 or {}).get("detail", {}).get("curve") or []
        if curve:
            # quote the budget with the strongest *lower bound*, not the largest point estimate
            best = max(curve, key=lambda c: c.get("paired_ci_low", 0.0))
            pooled = [c.get("pooled_auc") for c in curve if c.get("pooled_auc") is not None]
            pooled_txt = (
                f" Pooled across subjects the same attacker reaches AUC "
                f"{_f(max(pooled), 2)}, so the signal is strongest when a control for the very "
                "subject is available."
                if pooled
                else ""
            )
            anti.append(
                f"- **Semantic drift survives a full Tombstone erasure**: an attacker reaches "
                f"{_pct(best['paired'])} paired accuracy [95% CI {_pct(best.get('paired_ci_low'))}, "
                f"{_pct(best.get('paired_ci_high'))}] at a query budget of {best['budget']} "
                f"(n={int(best['n'])}) asking whether a subject was ever in the index, against a "
                f"same-cluster control matched on Top-K slots vacated.{pooled_txt} Tombstone "
                "measures this and reports it; it does not fix it (*Ghost Echoes*, arXiv 2608.20352)."
            )
    if u:
        for name in ("M1", "M2"):
            m = next((x for x in u["methods"] if x["name"] == name), None)
            if m:
                loss = m.get("mia", {}).get("loss", {})
                anti.append(
                    f"- **Approximate unlearning leaves residual extractability ({m['label']})**: canary {m['canary_extracted']}/{m['canary_total']} still extractable, MIA AUC {_f(loss.get('auc'), 2)} [{_f(loss.get('ci_low'), 2)},{_f(loss.get('ci_high'), 2)}]; exact shard retrain (M3) side by side above."
                )
    lines += anti or ["_No anti-results committed yet._"]
    lines.append("")
    if a:
        lines += attacks_section(a) + [""]
    OUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(regenerate(), file=sys.stderr)
