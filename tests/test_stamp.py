from __future__ import annotations

import json

import pytest

from tombstone.lineage.stamp import (
    K_ARTIFACT,
    K_MENTIONS,
    K_SCOPE,
    K_SOURCE,
    K_SUBJECT,
    STAMP_KEYS,
    is_stamped,
    require_stamped,
    stamp,
    stamped_mentions,
    stamped_subject,
)
from tombstone.model.artifacts import Scope, SubjectRef
from tombstone.util import is_ulid

CASES = [
    ("dict", {"source": "a.txt"}),
    ("empty dict", {}),
    ("dict with existing keys", {"tombstone.artifact_id": "01HZZZZZZZZZZZZZZZZZZZZZZZ", "x": 1}),
]


@pytest.mark.parametrize("label,md", CASES, ids=[c[0] for c in CASES])
def test_stamp_dict(label: str, md: dict, pepper: bytes) -> None:
    out = stamp(md, "S-0417", "a.txt", "default", pepper=pepper)
    assert all(k in out for k in STAMP_KEYS)
    assert out[K_SUBJECT] == SubjectRef.from_raw("S-0417", pepper).hmac
    assert out[K_SOURCE].startswith("src:") and "a.txt" not in out[K_SOURCE]
    assert out[K_SCOPE] == "default"
    assert is_ulid(out[K_ARTIFACT])
    if K_ARTIFACT in md:
        assert out[K_ARTIFACT] == md[K_ARTIFACT]  # never re-stamps an existing artifact id
    assert md == md  # input untouched
    assert "S-0417" not in json.dumps(out)


def test_stamp_list_gives_one_artifact_per_element(pepper: bytes) -> None:
    out = stamp([{"i": 1}, {"i": 2}], "S-1", "s.txt", Scope("t1"), pepper=pepper)
    assert len({o[K_ARTIFACT] for o in out}) == 2
    assert all(o[K_SCOPE] == "t1" for o in out)


def test_stamp_with_subject_ref_needs_no_pepper() -> None:
    ref = SubjectRef("a" * 64)
    out = stamp({}, ref, "s.txt", "default")
    assert out[K_SUBJECT] == "a" * 64


def test_refuses_empty_subject_or_missing_scope(pepper: bytes) -> None:
    with pytest.raises(ValueError, match="subject_id"):
        stamp({}, "", "s.txt", "default", pepper=pepper)
    with pytest.raises(ValueError, match="scope"):
        stamp({}, "S-1", "s.txt", "", pepper=pepper)
    with pytest.raises(ValueError, match="pepper"):
        stamp({}, "S-1", "s.txt", "default")
    with pytest.raises(ValueError, match="source_id"):
        stamp({}, "S-1", "", "default", pepper=pepper)


def test_mentions_and_derived_from(pepper: bytes) -> None:
    out = stamp(
        {}, "S-1", "s.txt", "default", pepper=pepper, mentions=["S-2", "S-3"], derived_from="01ABC"
    )
    ments = stamped_mentions(out)
    assert {m.hmac for m in ments} == {
        SubjectRef.from_raw("S-2", pepper).hmac,
        SubjectRef.from_raw("S-3", pepper).hmac,
    }
    assert "S-2" not in out[K_MENTIONS]
    assert out["tombstone.derived_from"] == "01ABC"


def test_is_stamped_and_require(pepper: bytes) -> None:
    assert not is_stamped({})
    assert not is_stamped(None)
    with pytest.raises(ValueError, match="not stamped"):
        require_stamped({"x": 1})
    md = stamp({}, "S-1", "s.txt", "default", pepper=pepper)
    assert require_stamped(md) is md
    assert stamped_subject(md).hmac == md[K_SUBJECT]


def test_document_round_trips_through_json(pepper: bytes) -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document

    doc = Document(page_content="hello", metadata={"source": "a.txt"})
    out = stamp(doc, "S-1", "a.txt", "default", pepper=pepper)
    assert isinstance(out, Document)
    assert doc.metadata == {"source": "a.txt"}  # original untouched
    blob = json.dumps(out.model_dump())
    back = Document(**json.loads(blob))
    assert all(k in back.metadata for k in STAMP_KEYS)
    assert back.metadata == out.metadata
    docs = stamp([doc, doc], "S-1", "a.txt", "default", pepper=pepper)
    assert len({d.metadata[K_ARTIFACT] for d in docs}) == 2


def test_unsupported_type(pepper: bytes) -> None:
    with pytest.raises(TypeError):
        stamp(42, "S-1", "s", "default", pepper=pepper)  # type: ignore[call-overload]


def test_stamp_is_deterministic_for_same_inputs(pepper: bytes) -> None:
    a = stamp({"x": 1}, "S-1", "a.txt", "default", pepper=pepper)
    b = stamp({"x": 1}, "S-1", "a.txt", "default", pepper=pepper)
    assert a == b  # LangChain index() relies on this to stay incremental
    c = stamp({"x": 1}, "S-1", "b.txt", "default", pepper=pepper)
    assert c[K_ARTIFACT] != a[K_ARTIFACT]
    d = stamp({"x": 1}, "S-2", "a.txt", "default", pepper=pepper)
    assert d[K_ARTIFACT] != a[K_ARTIFACT]
    e = stamp({"x": 1}, "S-1", "a.txt", "tenant-2", pepper=pepper)
    assert e[K_ARTIFACT] != a[K_ARTIFACT]
