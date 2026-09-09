"""Derive bench/thresholds.json from the latest committed results (9.3 nightly gate).

Thresholds are the measured value plus a margin, committed next to the numbers that set them;
loosening one later is a reviewed change. Run after a benchmark you are happy with:

    uv run python bench/set_thresholds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
OUT = ROOT / "bench" / "thresholds.json"


def main() -> int:
    thr: dict[str, dict[str, dict[str, float]]] = {"residue": {}, "unlearn": {}}
    r = RESULTS / "residue-latest.json"
    if r.is_file():
        for c in json.loads(r.read_text())["cells"]:
            key = f"{c['backend']}.{c['baseline']}"
            entry: dict[str, float] = {
                "min_logical_exclusion": round(max(0.0, c["logical_exclusion_rate"] - 0.05), 3),
                "max_recall_drop": 0.05,
            }
            if c["baseline"] == "B4" and c["physical_checked"]:
                entry["max_physical_residue"] = round(
                    min(1.0, c["physical_residue_rate"] + 0.05), 3
                )
            thr["residue"][key] = entry
    u = RESULTS / "unlearn-latest.json"
    if u.is_file():
        for m in json.loads(u.read_text())["methods"]:
            if m["name"] in {"M3", "M4"}:
                thr["unlearn"][m["name"]] = {
                    "max_canary_rate": round(min(1.0, m["canary_rate"] + 0.05), 3)
                }
    OUT.write_text(json.dumps(thr, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
