"""5.3: Ghost Echoes drift reproduced on our corpus (chroma, faiss), numbers committed by bench."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import _stores
from tombstone.verify.semantic import DriftProbe


@pytest.mark.parametrize("backend", ["chroma", "faiss"])
def test_drift_measurement_runs_and_reports_ci(backend: str, tmp_path: Path, pepper: bytes) -> None:
    _stores.skip_unless(backend)
    store = _stores.make_backend(backend, tmp_path)
    emb = _stores.embedder()
    docs = _stores.stamped_docs(
        pepper, n_subjects=8, per_subject=2
    )  # 80 chunks, clusters by subject
    with _stores.lineage_and_capture(tmp_path) as (_lineage, capture):
        texts = [x for x, _ in docs]
        recs = capture.prepare_embeds(
            store.name,
            emb.name,
            [f"k{i}" for i in range(len(docs))],
            emb.embed(texts),
            [m for _, m in docs],
            texts,
        )
        store.add(recs)
        target = recs[0].embed_node.ref()
        probe = DriftProbe(store, budget=5, seed=1)
        assert probe.record_before(target.store_key)
        store.reclaim([target])
        res = probe.after(target.store_key)
        assert res is not None
        for k in (
            "drift",
            "control",
            "drift_ci_low",
            "drift_ci_high",
            "control_ci_low",
            "control_ci_high",
            "query_budget",
        ):
            assert k in res
        assert res["query_budget"] == 5
        assert res["drift_ci_low"] <= res["drift"] <= res["drift_ci_high"]
        print(
            f"{backend}: drift {res['drift']:.4f} [{res['drift_ci_low']:.4f},{res['drift_ci_high']:.4f}] control {res['control']:.4f} above={res['above_control']}"
        )
    store.close()
