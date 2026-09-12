"""The Pinecone adapter: the first backend whose bytes nobody can check.

Two things are under test. The ordinary one is the adapter contract — it has to behave like
every other store. The one that matters is the *refusal*: a managed service must report
`{LOGICAL}` and say plainly that the stored bytes were not examined, because the failure this
project exists to prevent is a tool reporting a clean pass over a database it could not see
into.

These run against an in-process fake (`tests/_pinecone_fake.py`). That is honest about what it
proves: the adapter is correct against the client API as documented and against the service
behaviour the adapter is built around — keyword-only calls and eventual consistency. It is not
evidence about the live service, and `docs/adapters.md` says so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests import _pinecone_fake, _stores
from tests.test_adapter_suite import adapter_contract
from tombstone.errors import NotSupported
from tombstone.model.status import VerifyLevel

pytestmark = pytest.mark.skipif(
    not _stores.has_module("pinecone"), reason="pinecone client not installed"
)


def _store(monkeypatch: pytest.MonkeyPatch, lag: int = 2) -> Any:
    from tombstone.stores.pinecone import PineconeStore

    _pinecone_fake.install(monkeypatch, lag=lag)
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test-key")
    return PineconeStore("pinecone:kb", index="kb", embedding_model="hash-embed-64", dims=64)


def _records(tmp_path: Path, pepper: bytes, n: int = 3) -> Any:
    emb = _stores.embedder()
    docs = _stores.stamped_docs(pepper, n_subjects=n, per_subject=1)[:n]
    with _stores.lineage_and_capture(tmp_path) as (_lineage, capture):
        texts = [t for t, _ in docs]
        return capture.prepare_embeds(
            "pinecone:kb",
            emb.name,
            [f"k{i}" for i in range(n)],
            emb.embed(texts),
            [m for _, m in docs],
            texts,
        )


def test_it_passes_the_same_contract_as_every_other_store(
    tmp_path: Path, pepper: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(monkeypatch)
    recs = _records(tmp_path, pepper)
    refs = [r.embed_node.ref() for r in recs]
    pending = {r.key: r for r in recs}
    adapter_contract(store, lambda r: store.add([pending[r.store_key]]), refs)


def test_it_refuses_to_pretend_it_can_see_the_bytes(
    tmp_path: Path, pepper: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole reason this adapter is interesting. It must not return `found=False` from a
    physical probe — "I looked and it is gone" and "I cannot look" are different answers, and
    only one of them is true here."""
    store = _store(monkeypatch)
    recs = _records(tmp_path, pepper)
    store.add(recs)
    ref = recs[0].embed_node.ref()

    assert store.capabilities == frozenset({VerifyLevel.LOGICAL})
    assert VerifyLevel.PHYSICAL not in store.capabilities
    assert VerifyLevel.SEMANTIC not in store.capabilities

    with pytest.raises(NotSupported) as e:
        store.probe_physical(ref)
    msg = str(e.value)
    assert "managed service" in msg
    assert "does not claim" in msg or "cannot" in msg
    assert store.persisted_files() == []
    store.close()


def test_suppression_waits_for_the_service_to_agree(
    tmp_path: Path, pepper: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Suppression that has not propagated is not suppression. With a lagging service the flag
    is invisible for several reads; the adapter must not return until it is visible, or Hard
    Rule 5 (suppressed before reclaimed) is only true on paper."""
    store = _store(monkeypatch, lag=3)
    recs = _records(tmp_path, pepper)
    store.add(recs)
    refs = [r.embed_node.ref() for r in recs]
    from tombstone.stores.base import ProbeSet

    ps = ProbeSet(refs[0].artifact_id)
    assert store.probe_logical(refs[0], ps).found

    store.suppress(refs[:1])
    # by the time suppress returns, every query path must already miss it
    assert not store.probe_logical(refs[0], ps).found
    assert store.probe_logical(refs[1], ps).found
    store.close()


def test_reclaim_reports_the_settle_time_rather_than_assuming_instant(
    tmp_path: Path, pepper: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(monkeypatch, lag=3)
    recs = _records(tmp_path, pepper)
    store.add(recs)
    refs = [r.embed_node.ref() for r in recs]

    res = store.reclaim(refs[:1])
    assert not res.noop
    assert "delete" in res.method
    assert "not observable" in res.method, "the method string must not imply a byte-level result"
    assert res.measurement["deleted"] == 1.0
    assert res.measurement["still_returned"] == 0.0
    assert "settle_s" in res.measurement
    # and doing it again is a no-op, as the contract requires
    assert store.reclaim(refs[:1]).noop
    store.close()


def test_a_slow_service_is_reported_not_hidden(
    tmp_path: Path, pepper: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the delete has not become visible inside the timeout, the adapter says so instead of
    waiting forever or claiming success."""
    store = _store(monkeypatch, lag=10_000)  # never converges within the test's patience
    store.settle_timeout_s = 0.5
    recs = _records(tmp_path, pepper)
    store.add(recs)
    refs = [r.embed_node.ref() for r in recs]

    res = store.reclaim(refs[:1])
    assert not res.noop
    assert res.measurement["still_returned"] == 1.0
    assert "eventually consistent" in res.detail
    store.close()


def test_the_api_key_never_reaches_anything_persisted(
    tmp_path: Path, pepper: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hard Rule 7's neighbour: a receipt or a log carrying the operator's Pinecone key would be
    a credential leak shipped in an audit artefact."""
    store = _store(monkeypatch)
    blob = " ".join(
        [
            store.version(),
            store.physical_unsupported_reason(),
            repr(store.reclaim([])),
            store.name,
        ]
    )
    assert "pc-test-key" not in blob
    store.close()


def test_a_missing_key_says_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    from tombstone.stores.pinecone import PineconeStore

    _pinecone_fake.install(monkeypatch)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(NotSupported) as e:
        PineconeStore("pinecone:kb", index="kb")
    assert "PINECONE_API_KEY" in str(e.value)


def test_config_accepts_a_pinecone_store() -> None:
    """It has to be reachable from tombstone.yaml, not just importable."""
    from tombstone.config import StoreConfig

    sc = StoreConfig(name="pinecone:kb", kind="pinecone", index="kb", embedding="hash-embed-64")
    assert sc.kind == "pinecone" and sc.index == "kb"
    with pytest.raises(Exception):  # index is required  # noqa: B017
        StoreConfig(name="bad", kind="pinecone", embedding="hash-embed-64")
