"""Inject the generated, traceable blocks into README.md between markers (Hard Rule 12).

    uv run python bench/readme_numbers.py

Blocks: the two headline numbers (6.6), the results table, the Ghost Echoes detection curve
(7.5), and Demo 1 from docs/demo.md. Everything inside ``<!-- measured:start -->`` … ``end``
is checked by ``python -m tombstone._checks readme-numbers`` against bench/results/*.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
README = ROOT / "README.md"
DEMO = ROOT / "docs" / "demo.md"


def _load(kind: str) -> dict[str, Any] | None:
    p = RESULTS / f"{kind}-latest.json"
    return json.loads(p.read_text()) if p.is_file() else None


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def headline_block() -> str:
    r, u, a = _load("residue"), _load("unlearn"), _load("attacks")
    lines = ["<!-- measured:start -->"]
    if r:
        b0 = [c for c in r["cells"] if c["baseline"] == "B0"]
        checked = [c for c in b0 if c["physical_checked"]]
        rate = (
            sum(c["physical_residue_rate"] for c in checked) / max(1, len(checked))
            if checked
            else None
        )
        per = ", ".join(f"{c['backend']} {_pct(c['physical_residue_rate'])}" for c in checked)
        b4 = [c for c in r["cells"] if c["baseline"] == "B4" and c["physical_checked"]]
        rate4 = sum(c["physical_residue_rate"] for c in b4) / max(1, len(b4)) if b4 else None
        own4 = sum(c.get("own_record_rate") or 0.0 for c in b4) / max(1, len(b4)) if b4 else None
        lines.append(
            f"1. **After a native `delete()`, {_pct(rate)} of a subject's vectors are still physically recoverable** "
            f"from the index files across the checked backends ({per}; {r['corpus']['subjects']} subjects each), "
            f"while every one of them is logically gone ({_pct(sum(c['logical_exclusion_rate'] for c in b0) / len(b0))} exclusion). "
            f"After `tombstone erase`, {_pct(own4)} of the subjects' own records remain and {_pct(rate4)} of vectors still have "
            "byte-identical copies in the files, all belonging to other subjects' boilerplate and reported UNVERIFIED(duplicate content), never VERIFIED. "
            "Source: `bench/results/residue-latest.json`, command `uv run python bench/residue/run_residue.py --all`."
        )
    if u:

        def fmt(m: dict[str, Any]) -> str:
            loss = m.get("mia", {}).get("loss", {})
            return (
                f"canary extraction {m['canary_extracted']}/{m['canary_total']}, MIA AUC "
                f"{loss.get('auc', float('nan')):.2f} [{loss.get('ci_low', float('nan')):.2f},{loss.get('ci_high', float('nan')):.2f}], "
                f"held-out perplexity {m['holdout_ppl']:.1f}"
            )

        m = {x["name"]: x for x in u["methods"]}
        if "M3" in m and "M0" in m:
            parts = [f"exact shard retrain: {fmt(m['M3'])}"]
            for k, label in (("M1", "NPO"), ("M2", "gradient difference")):
                if k in m:
                    parts.append(f"{label}: {fmt(m[k])}")
            lines.append(
                f"2. **Exact shard unlearning vs approximate** on `{u['model']}` ({u['subjects']} subjects): before, {fmt(m['M0'])}; "
                + "; ".join(parts)
                + ". Source: `bench/results/unlearn-latest.json`, command `uv run python bench/unlearn/run_unlearn.py --all`."
            )
    if a:
        s75 = next((s for s in a["strategies"] if s["id"] == "7.5"), None)
        if s75 and s75.get("detail", {}).get("curve"):
            curve = s75["detail"]["curve"]
            lines.append(
                "3. **The layer nobody can erase**: after a full Tombstone erasure, an attacker estimating "
                '"was this subject ever here?" from retrieval-context drift (*Ghost Echoes* protocol) reaches paired-comparison accuracy '
                + ", ".join(f"{_pct(c['paired'])} at budget {c['budget']}" for c in curve)
                + ". Tombstone measures and reports this; it does not fix it. Source: `bench/results/attacks-latest.json`."
            )
    if len(lines) == 1:
        lines.append("_Benchmarks have not been run yet; no numbers are claimed._")
    lines.append("<!-- measured:end -->")
    return "\n".join(lines)


def results_table() -> str:
    r = _load("residue")
    if not r:
        return ""
    lines = [
        "<!-- measured:start -->",
        "| backend | method | logical exclusion | physical residue | Recall@5 before → after | wall/erasure |",
        "|---|---|---|---|---|---|",
    ]
    for c in r["cells"]:
        phys = _pct(c["physical_residue_rate"]) if c["physical_checked"] else "UNVERIFIED"
        lines.append(
            f"| {c['backend']} | {c['baseline']} {c['label']} | {_pct(c['logical_exclusion_rate'])} | {phys} | {_pct(c['recall_at_5_before'])} → {_pct(c['recall_at_5_after'])} | {c['wall_s_mean']:.2f}s |"
        )
    lines.append("<!-- measured:end -->")
    return "\n".join(lines)


def demo1_block() -> str:
    if not DEMO.is_file():
        return "_Run `uv run python bench/demo.py` to generate the demos._"
    text = DEMO.read_text()
    m = re.search(r"## Demo 1.*?```text\n(.*?)```", text, re.S)
    return "```text\n" + m.group(1) + "```" if m else ""


def inject(readme: str, name: str, block: str) -> str:
    start, end = f"<!-- generated:{name}:start -->", f"<!-- generated:{name}:end -->"
    if start not in readme:
        return readme
    pre, rest = readme.split(start, 1)
    _, post = rest.split(end, 1)
    return f"{pre}{start}\n{block}\n{end}{post}"


def main() -> int:
    readme = README.read_text()
    readme = inject(readme, "headline", headline_block())
    readme = inject(readme, "results", results_table())
    readme = inject(readme, "demo1", demo1_block())
    README.write_text(readme)
    print("README blocks regenerated", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
