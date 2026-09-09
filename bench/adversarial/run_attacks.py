"""Phase 7 — break your own eraser. One strategy per file under ``strategies/``; each returns a
row {id, name, survives, rate, rate_text, mitigation, detail}. Rates are measured, never
assumed; 7.1, 7.4 and 7.5 are non-zero by construction.

    uv run python bench/adversarial/run_attacks.py --all
    uv run python bench/adversarial/run_attacks.py --only 7.1,7.4

Writes bench/results/attacks-<ts>.json and regenerates RESULTS.md.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import cost_add, save_results

STRATEGIES = [
    ("7.1", "s71_lineage_gap"),
    ("7.2", "s72_derived_content"),
    ("7.3", "s73_paraphrased_cache"),
    ("7.4", "s74_third_party"),
    ("7.5", "s75_ghost_echo_budget"),
    ("7.6", "s76_relearning"),
    ("7.7", "s77_backup_replica"),
    ("7.8", "s78_suppression_race"),
    ("7.9", "s79_concurrent_shared"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--subjects", type=int, default=20)
    ns = ap.parse_args(argv)
    wanted = set(ns.only.split(",")) if ns.only else {s for s, _ in STRATEGIES}
    rows: list[dict[str, Any]] = []
    t_all = time.time()
    for sid, mod in STRATEGIES:
        if sid not in wanted:
            continue
        print(f"[attacks] {sid} {mod} …", file=sys.stderr, flush=True)
        t0 = time.time()
        m = importlib.import_module(f"adversarial.strategies.{mod}")
        try:
            row = m.run(ns.subjects)
        except Exception as e:
            row = {
                "id": sid,
                "name": mod,
                "survives": "attack did not run",
                "rate": None,
                "rate_text": f"ERROR {type(e).__name__}: {e}",
                "mitigation": "",
                "detail": {},
            }
        row["wall_s"] = round(time.time() - t0, 1)
        rows.append(row)
        print(
            f"[attacks] {sid}: {row['rate_text']} ({row['wall_s']}s)", file=sys.stderr, flush=True
        )
    path = save_results("attacks", {"strategies": rows, "subjects": ns.subjects})
    cost_add("attacks-bench", time.time() - t_all)
    print(f"wrote {path}", file=sys.stderr)
    from report import regenerate

    regenerate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
