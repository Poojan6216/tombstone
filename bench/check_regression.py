"""Nightly gate: fail the build if a headline metric regresses beyond bench/thresholds.json.

The benchmark is a test, not a marketing artefact. Thresholds are committed alongside the
numbers that set them; loosening one is a reviewed change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
THRESHOLDS = ROOT / "bench" / "thresholds.json"


def main() -> int:
    if not THRESHOLDS.is_file():
        print("no thresholds committed; nothing to check", file=sys.stderr)
        return 0
    thr = json.loads(THRESHOLDS.read_text())
    problems: list[str] = []
    r = (
        json.loads((RESULTS / "residue-latest.json").read_text())
        if (RESULTS / "residue-latest.json").is_file()
        else None
    )
    if r:
        for c in r["cells"]:
            key = f"{c['backend']}.{c['baseline']}"
            t = thr.get("residue", {}).get(key)
            if not t:
                continue
            if (
                c["baseline"] == "B4"
                and c["physical_checked"]
                and c["physical_residue_rate"] > t.get("max_physical_residue", 0.0)
            ):
                problems.append(
                    f"{key}: physical residue {c['physical_residue_rate']:.3f} > {t['max_physical_residue']}"
                )
            if c["logical_exclusion_rate"] < t.get("min_logical_exclusion", 0.0):
                problems.append(
                    f"{key}: logical exclusion {c['logical_exclusion_rate']:.3f} < {t['min_logical_exclusion']}"
                )
            if c["recall_at_5_after"] < c["recall_at_5_before"] - t.get("max_recall_drop", 1.0):
                problems.append(
                    f"{key}: recall@5 dropped {c['recall_at_5_before']:.3f} → {c['recall_at_5_after']:.3f}"
                )
    u = (
        json.loads((RESULTS / "unlearn-latest.json").read_text())
        if (RESULTS / "unlearn-latest.json").is_file()
        else None
    )
    if u:
        for m in u["methods"]:
            t = thr.get("unlearn", {}).get(m["name"])
            if not t:
                continue
            if m["canary_rate"] > t.get("max_canary_rate", 1.0):
                problems.append(
                    f"{m['name']}: canary rate {m['canary_rate']:.3f} > {t['max_canary_rate']}"
                )
    for p in problems:
        print("REGRESSION " + p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
