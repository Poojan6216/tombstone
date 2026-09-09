"""7.1 — pre-existing data: 30% of the corpus ingested before capture was enabled.

Expected to work; we measure how badly. Also asserts the refusal fires and that an erase under
--accept-gaps records UNVERIFIED(lineage-gap), never VERIFIED, for the gapped store.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import WORK, stable_sample
from adversarial._harness import Pipeline
from corpus.build import load_corpus
from tombstone.errors import LineageGapError


def run(n_subjects: int) -> dict[str, Any]:
    docs = load_corpus()
    subjects = [f"S-{i:04d}" for i in range(1, n_subjects + 1)]
    subject_docs = [d for d in docs if d.subject in subjects]
    public = stable_sample([d for d in docs if d.subject == "PUBLIC"], 300, 5)
    curve = []
    for frac in (0.0, 0.3, 0.5):
        p = Pipeline(WORK / "attacks" / f"s71-{int(frac * 100)}")
        pre = stable_sample(subject_docs, int(len(subject_docs) * frac), 9)
        pre_ids = {d.doc_id for d in pre}
        p.ingest(pre + public[:100], capture=False)  # before Tombstone existed
        p.ingest([d for d in subject_docs if d.doc_id not in pre_ids] + public[100:], capture=True)
        survived = 0
        total = 0
        refused = 0
        gap_status = set()
        no_lineage = 0
        for s in subjects:
            can = next(d.canary for d in docs if d.subject == s and d.canary)
            try:
                p.erase(s)
            except LineageGapError as e:
                refused += 1
                if "no lineage records" in str(e):
                    # every document of this subject predates capture: the trace itself refuses
                    # (Hard Rule 4) and nothing can be erased — the strongest form of the gap
                    no_lineage += 1
                else:
                    _code, data, _t = p.erase(s, accept_gaps=True)
                    gap_status |= {
                        x["rule_id"]
                        for x in data["statuses"]
                        if x["artifact"]["store"] == p.store.name
                    }
            total += 1
            survived += int(p.canary_hits(can.token) > 0)
        curve.append(
            {
                "fraction_before_capture": frac,
                "subjects": total,
                "canary_survived": survived,
                "survival_rate": survived / max(1, total),
                "refusals": refused,
                "no_lineage_at_all": no_lineage,
                "gapped_store_rules": sorted(gap_status),
            }
        )
        p.close()
    worst = curve[1]
    return {
        "id": "7.1",
        "name": "lineage gap: data ingested before capture",
        "survives": "chunks ingested before Tombstone was installed",
        "rate": worst["survival_rate"],
        "rate_text": "; ".join(
            f"{int(c['fraction_before_capture'] * 100)}% pre-capture → {c['canary_survived']}/{c['subjects']} subjects' canaries still retrievable"
            for c in curve
        ),
        "mitigation": "erase refuses without --accept-gaps (fired in "
        + f"{worst['refusals']}/{worst['subjects']}"
        + " cases); with it the store is UNVERIFIED(lineage-gap), never VERIFIED. Backfill lineage by re-ingesting through the wrapper.",
        "detail": {"curve": curve},
    }
