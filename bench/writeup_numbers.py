"""9.5: fill docs/writeup.md from the committed result files.

The post is prose, but its numbers are measurements, so it is generated the way RESULTS.md is
rather than hand-edited — otherwise the day a benchmark is re-run the post quietly becomes wrong.
Every placeholder must resolve; an unresolved one, or a rate line that no longer matches the
shape this reads, fails the build rather than printing something plausible.

    uv run python bench/writeup_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
TMPL = ROOT / "bench" / "writeup.tmpl.md"
OUT = ROOT / "docs" / "writeup.md"


def _load(kind: str) -> dict[str, Any]:
    p = RESULTS / f"{kind}-latest.json"
    if not p.is_file():
        raise SystemExit(f"writeup: missing {p}; run the benchmark first")
    return dict(json.loads(p.read_text(encoding="utf-8")))


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _rng(vals: list[float]) -> str:
    lo, hi = min(vals), max(vals)
    return _pct(lo) if abs(hi - lo) < 1e-9 else f"{_pct(lo)}–{_pct(hi)}"


def _grab(text: str, pattern: str, what: str) -> str:
    """One capture group, or fail: a post that cannot find its own number must not invent one."""
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"writeup: could not read {what} from {text[:160]!r}")
    return m.group(1)


def _approx_caveat(meth: dict[str, Any]) -> str:
    """NPO and gradient difference act on the unsharded adapter. If that adapter had collapsed,
    the post must not state their numbers as findings — the same test the report makes."""
    flat = meth.get("M0-unsharded", {}).get("holdout_ppl")
    ens = meth.get("M0", {}).get("holdout_ppl")
    if not (flat and ens and flat > 20.0 * ens):
        return ""
    return (
        "\n\n**On the two approximate methods, this run has no result.** Both are applied to the "
        f"unsharded adapter, whose held-out perplexity is {flat:.0f} against {ens:.1f} for the shard "
        "ensemble: it had collapsed before any unlearning ran, so their canary counts and the "
        "resurfacing above describe damage to a broken model rather than the methods. The figures "
        "are shown because they were measured. The exact-retrain numbers are unaffected — they are "
        "measured on the shard ensemble."
    )


def values() -> dict[str, str]:
    r, u, a, d = _load("residue"), _load("unlearn"), _load("attacks"), _load("demo")
    meth = {m["name"]: m for m in u["methods"]}
    strat = {s["id"]: s for s in a["strategies"]}
    b0 = [c for c in r["cells"] if c["baseline"] == "B0" and c["physical_checked"]]

    def canary(name: str) -> str:
        m = meth[name]
        return f"{m['canary_extracted']}/{m['canary_total']}"

    def auc(name: str, key: str = "auc") -> str:
        return f"{meth[name].get('mia', {}).get('loss', {}).get(key, float('nan')):.2f}"

    def relearn(method_sub: str, steps: int) -> str:
        for row in u.get("relearn", []):
            if method_sub in row["method"].lower() and int(row["steps"]) == steps:
                return f"{row['canary_extracted']}/{row['canary_total']}"
        raise SystemExit(f"writeup: no relearn row for {method_sub} at {steps} steps")

    curve = {int(c["budget"]): c for c in strat["7.5"]["detail"]["curve"]}
    chaos = _grab(
        (ROOT / "tests" / "test_erase.py").read_text(encoding="utf-8"),
        r"rng\.sample\(range\([^)]*\),\s*(\d+)\)",
        "the number of chaos kill points",
    )
    return {
        "demo1_artifacts": str(d["demo1"]["artifacts"]),
        "demo1_recoverable": str(d["demo1"]["recoverable"]),
        "residue_subjects": str(r["corpus"]["subjects"]),
        "b0_residue_range": _rng([c["physical_residue_rate"] for c in b0]),
        "b0_logical": _rng([c["logical_exclusion_rate"] for c in b0]),
        "chaos_points": chaos,
        "unlearn_subjects": str(u["subjects"]),
        "m0_canary": canary("M0"),
        "m3_canary": canary("M3"),
        "m1_canary": canary("M1"),
        "m2_canary": canary("M2"),
        "m0_auc": auc("M0"),
        "m3_auc": auc("M3"),
        "m1_auc": auc("M1"),
        "m2_auc": auc("M2"),
        "m3_lo": auc("M3", "ci_low"),
        "m3_hi": auc("M3", "ci_high"),
        "m1_lo": auc("M1", "ci_low"),
        "m1_hi": auc("M1", "ci_high"),
        "ppl_delta": f"{meth['M3']['holdout_ppl'] - meth['M0']['holdout_ppl']:+.1f} "
        f"({meth['M0']['holdout_ppl']:.1f} → {meth['M3']['holdout_ppl']:.1f})",
        "approx_caveat": _approx_caveat(meth),
        "relearn_steps": "50",
        "relearn_npo": relearn("npo", 50),
        "relearn_exact": relearn("exact", 50),
        "drift_acc5": _pct(curve[5]["paired"]),
        "drift_acc40": _pct(curve[40]["paired"]),
        "a71": _grab(strat["7.1"]["rate_text"], r"30% pre-capture → (\S+) subjects", "7.1"),
        "a72": _grab(strat["7.2"]["rate_text"], r"unstamped summaries: (\S+) canaries", "7.2"),
        "a72m": _grab(strat["7.2"]["rate_text"], r"derived_from stamping: (\S+)", "7.2 mitigated"),
        "a73": _grab(strat["7.3"]["rate_text"], r"purge_k=0: (\S+) subjects", "7.3"),
        "a74": _grab(strat["7.4"]["rate_text"], r"^(\S+) subjects", "7.4"),
        "a77": _grab(strat["7.7"]["rate_text"], r"Chroma dir: (\S+) subjects", "7.7"),
        "a78_lat": _grab(strat["7.8"]["rate_text"], r"latency median (\d+) ms", "7.8 latency"),
    }


def main() -> int:
    text = TMPL.read_text(encoding="utf-8")
    vals = values()
    for k, v in vals.items():
        text = text.replace("{{" + k + "}}", v)
    left = sorted(set(re.findall(r"\{\{(\w+)\}\}", text)))
    if left:
        raise SystemExit(f"writeup: unresolved placeholders {left}")
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(vals)} measured values)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
