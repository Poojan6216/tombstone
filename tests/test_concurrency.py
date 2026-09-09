"""8.4: concurrent sagas produce a valid chain or clean failures; 7.9 property test over shared
chunks. Run repeatedly in CI (the workflow loops this file 100×)."""

from __future__ import annotations

import random
import threading
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.conftest import requires_faiss, requires_langchain
from tombstone.commands.erase import run_erase
from tombstone.commands.trace import run_trace
from tombstone.erase.journal import Journal
from tombstone.errors import LockTimeout, SagaError
from tombstone.receipt.ledger import Ledger

pytestmark = [requires_langchain, requires_faiss]


def test_twenty_concurrent_sagas_valid_chain_or_clean_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests import _pipeline

    monkeypatch.chdir(tmp_path)
    h = _pipeline.build(tmp_path, ["faiss"], rag_subjects=())
    rt = h["rt"]
    from tests._corpus import subjects

    subs = subjects()  # 8 subjects; run 20 sagas: 8 real + 12 retries/duplicates racing
    traces = {s: run_trace(rt, s, with_store_gaps=False)[0].trace_id for s in subs}
    outcomes: list[str] = []
    lock = threading.Lock()

    def go(i: int) -> None:
        s = subs[i % len(subs)]
        try:
            code, _t, _d = run_erase(rt, traces[s], f"c-{i}", confirm=True, retry=i >= len(subs))
            res = f"ok:{code}"
        except (LockTimeout, SagaError) as e:
            res = f"clean:{type(e).__name__}"
        except Exception as e:  # noqa: BLE001
            res = f"UNCLEAN:{type(e).__name__}:{e}"
        with lock:
            outcomes.append(res)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(20)]
    random.Random(84).shuffle(threads)
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not [o for o in outcomes if o.startswith("UNCLEAN")], outcomes
    assert Journal(rt.inst.journal_path).verify() > 0
    assert Ledger(rt.inst.ledger_path).verify() >= 1
    assert not Journal(rt.inst.journal_path).open_sagas()
    rt.close()


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(overlap=st.integers(min_value=0, max_value=3), order=st.booleans())
def test_shared_chunks_final_state_is_set_difference(
    tmp_path_factory: pytest.TempPathFactory, overlap: int, order: bool
) -> None:
    """Two subjects share `overlap` identical chunks; erased concurrently → exactly the difference."""
    from langchain_core.documents import Document

    from tombstone.commands.init import run_init
    from tombstone.integrations.langchain import TombstoneVectorStore
    from tombstone.lineage.stamp import stamp
    from tombstone.registry import Runtime

    root = tmp_path_factory.mktemp("shared")
    run_init(root)
    cfg = root / "tombstone.yaml"
    cfg.write_text(
        cfg.read_text().replace(
            "stores:\n",
            'stores:\n  - { name: "faiss:kb", kind: faiss, path: ./kb.index, embedding: hash-embed-64 }\n',
        )
    )
    rt = Runtime.shared(cfg)
    vs = TombstoneVectorStore.from_config("faiss:kb", config=cfg)
    pepper = rt.pepper()
    shared_texts = [f"shared boilerplate paragraph {i}" for i in range(overlap)]
    own = {
        "A": [f"only A text {i}" for i in range(3)],
        "B": [f"only B text {i}" for i in range(3)],
        "C": [f"only C text {i}" for i in range(2)],
    }
    from tombstone.util import derived_ulid

    for s, texts in own.items():
        docs = [
            stamp(
                Document(page_content=t, metadata={"source": f"{s}-{i}"}),
                s,
                f"{s}-{i}",
                "default",
                pepper=pepper,
            )
            for i, t in enumerate(texts)
        ]
        if s in {"A", "B"}:
            # genuinely shared chunks: one CHUNK node under both subjects (explicit chunk id)
            for i, t in enumerate(shared_texts):
                md = {"source": f"shared-{i}", "tombstone.chunk_id": derived_ulid("shared", str(i))}
                docs.append(
                    stamp(
                        Document(page_content=t, metadata=md),
                        s,
                        f"shared-{i}",
                        "default",
                        pepper=pepper,
                    )
                )
        vs.add_documents(docs)

    ta = run_trace(rt, "A", with_store_gaps=False)[0]
    tb = run_trace(rt, "B", with_store_gaps=False)[0]
    if overlap:
        assert ta.shared and tb.shared
    errors: list[str] = []

    def go(tid: str, reason: str) -> None:
        try:
            run_erase(rt, tid, reason, confirm=True)
        except (LockTimeout, SagaError) as e:
            errors.append(f"{reason}:{type(e).__name__}")

    t1 = threading.Thread(target=go, args=(ta.trace_id, "A"))
    t2 = threading.Thread(target=go, args=(tb.trace_id, "B"))
    first, second = (t1, t2) if order else (t2, t1)
    first.start()
    second.start()
    first.join()
    second.join()
    remaining_docs = {d.page_content for d in vs.similarity_search("text", k=50)}
    # C's data untouched; A's and B's (including shared) gone if both sagas ran
    assert set(own["C"]) <= remaining_docs
    if not errors:
        assert not (set(own["A"]) | set(own["B"]) | set(shared_texts)) & remaining_docs
    assert Journal(rt.inst.journal_path).verify() > 0
    rt.close()
