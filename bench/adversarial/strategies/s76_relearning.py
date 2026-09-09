"""7.6 — relearning attack: reported from the unlearning benchmark (bench/results/unlearn-latest.json)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import RESULTS


def run(n_subjects: int) -> dict[str, Any]:
    p = RESULTS / "unlearn-latest.json"
    if not p.is_file():
        return {
            "id": "7.6",
            "name": "relearning attack on unlearning",
            "survives": "forgotten canaries resurface after light continued training",
            "rate": None,
            "rate_text": "not run yet: run bench/unlearn/run_unlearn.py first",
            "mitigation": "exact shard retrain (M3): the data is not in the weights",
            "detail": {},
        }
    u = json.loads(p.read_text())
    rows = u.get("relearn", [])
    text = "; ".join(
        f"{r['method']} +{r['steps']} steps: {r['canary_extracted']}/{r['canary_total']}"
        for r in rows
    )
    worst = max(
        (
            r["canary_extracted"] / max(1, r["canary_total"])
            for r in rows
            if "exact" not in r["method"]
        ),
        default=None,
    )
    return {
        "id": "7.6",
        "name": "relearning attack on approximate unlearning",
        "survives": "canaries resurface after light continued training on unrelated data",
        "rate": worst,
        "rate_text": text or "no relearn rows",
        "mitigation": "exact shard retrain (M3) — the rows above labelled 'exact' show whether it resurfaces",
        "detail": {"rows": rows},
    }
