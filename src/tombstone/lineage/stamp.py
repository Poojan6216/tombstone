"""Stamping: attach subject, source, scope and an artifact id to a document at ingest.

Works on a plain ``dict`` of metadata, a ``langchain_core.documents.Document``, or a list of
either. The raw subject id is hashed immediately (Hard Rule 7); the raw source id is hashed too,
so nothing personal ever reaches a lineage row.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar, overload

from tombstone.model.artifacts import Scope, SubjectRef
from tombstone.util import derived_ulid, sha256_hex

K_SUBJECT = "tombstone.subject"
K_SOURCE = "tombstone.source"
K_SCOPE = "tombstone.scope"
K_ARTIFACT = "tombstone.artifact_id"
K_MENTIONS = "tombstone.mentions"
K_DERIVED_FROM = "tombstone.derived_from"
K_CHUNK = "tombstone.chunk_id"
K_EMBED = "tombstone.embed_id"
K_SUPPRESSED = "tombstone.suppressed"
STAMP_KEYS = (K_SUBJECT, K_SOURCE, K_SCOPE, K_ARTIFACT)

T = TypeVar("T")


def source_key(raw_source_id: str) -> str:
    """The lineage store_key for a source: a hash, never the raw path/id."""
    return "src:" + sha256_hex(raw_source_id)[:32]


def source_artifact_id(
    scope: Scope, subject: SubjectRef, raw_source_id: str, position: int = 0
) -> str:
    """Deterministic SOURCE artifact id. Re-stamping the same document yields the same id, so
    LangChain's ``index()`` sees an unchanged document hash and stays incremental."""
    return derived_ulid(
        "source", scope.tenant, subject.hmac, source_key(raw_source_id), str(position)
    )


def _stamp_dict(
    md: dict[str, Any],
    subject: SubjectRef,
    raw_source_id: str,
    scope: Scope,
    mentions: Sequence[SubjectRef],
    derived_from: str | None,
    position: int = 0,
) -> dict[str, Any]:
    out = dict(md)
    out[K_SUBJECT] = subject.hmac
    out[K_SOURCE] = source_key(raw_source_id)
    out[K_SCOPE] = scope.tenant
    out.setdefault(K_ARTIFACT, source_artifact_id(scope, subject, raw_source_id, position))
    if mentions:
        out[K_MENTIONS] = ",".join(sorted(m.hmac for m in mentions))
    if derived_from:
        out[K_DERIVED_FROM] = derived_from
    return out


def _is_document(obj: object) -> bool:
    return hasattr(obj, "page_content") and hasattr(obj, "metadata")


@overload
def stamp(
    target: dict[str, Any],
    subject_id: str | SubjectRef,
    source_id: str,
    scope: str | Scope,
    *,
    pepper: bytes | None = None,
    mentions: Sequence[str | SubjectRef] = (),
    derived_from: str | None = None,
) -> dict[str, Any]: ...


@overload
def stamp(
    target: list[T],
    subject_id: str | SubjectRef,
    source_id: str,
    scope: str | Scope,
    *,
    pepper: bytes | None = None,
    mentions: Sequence[str | SubjectRef] = (),
    derived_from: str | None = None,
) -> list[T]: ...


@overload
def stamp(
    target: T,
    subject_id: str | SubjectRef,
    source_id: str,
    scope: str | Scope,
    *,
    pepper: bytes | None = None,
    mentions: Sequence[str | SubjectRef] = (),
    derived_from: str | None = None,
) -> T: ...


def stamp(
    target: Any,
    subject_id: str | SubjectRef,
    source_id: str,
    scope: str | Scope,
    *,
    pepper: bytes | None = None,
    mentions: Sequence[str | SubjectRef] = (),
    derived_from: str | None = None,
) -> Any:
    """Return a stamped copy of ``target``.

    ``subject_id`` may be a raw id (then ``pepper`` is required) or an already-hashed
    ``SubjectRef``. ``mentions`` are other subjects the app says this document names — recorded,
    never searched for. ``derived_from`` is the artifact id of the document this one was derived
    from (a summary, a translation) so the edge is not lost.
    """
    if isinstance(subject_id, SubjectRef):
        subject = subject_id
    else:
        if not subject_id:
            raise ValueError("subject_id must be non-empty")
        if pepper is None:
            raise ValueError("pepper is required to stamp with a raw subject id")
        subject = SubjectRef.from_raw(subject_id, pepper)
    if not source_id:
        raise ValueError("source_id must be non-empty")
    if isinstance(scope, Scope):
        sc = scope
    else:
        if not scope:
            raise ValueError("scope is required (use 'default' if single-tenant)")
        sc = Scope(scope)
    ments: list[SubjectRef] = []
    for m in mentions:
        if isinstance(m, SubjectRef):
            ments.append(m)
        else:
            if pepper is None:
                raise ValueError("pepper is required to stamp raw mention ids")
            ments.append(SubjectRef.from_raw(m, pepper))

    if isinstance(target, list):
        # One artifact id per element (position-derived); each element is its own document.
        return [
            _stamp_one(t, subject, source_id, sc, ments, derived_from, i)
            for i, t in enumerate(target)
        ]
    return _stamp_one(target, subject, source_id, sc, ments, derived_from, 0)


def _stamp_one(
    target: Any,
    subject: SubjectRef,
    source_id: str,
    sc: Scope,
    ments: list[SubjectRef],
    derived_from: str | None,
    position: int,
) -> Any:
    if isinstance(target, dict):
        return _stamp_dict(target, subject, source_id, sc, ments, derived_from, position)
    if _is_document(target):
        md = _stamp_dict(
            dict(target.metadata or {}), subject, source_id, sc, ments, derived_from, position
        )
        return target.model_copy(update={"metadata": md})
    raise TypeError(f"cannot stamp {type(target).__name__}; expected dict, Document, or list")


def is_stamped(md: dict[str, Any] | None) -> bool:
    if not md:
        return False
    return all(k in md for k in STAMP_KEYS)


def require_stamped(md: dict[str, Any] | None, what: str = "document") -> dict[str, Any]:
    if md is None or not is_stamped(md):
        missing = [k for k in STAMP_KEYS if md is None or k not in md]
        raise ValueError(
            f"{what} is not stamped (missing {', '.join(missing)}); call "
            "tombstone.lineage.stamp.stamp(doc, subject_id, source_id, scope, pepper=...) at ingest"
        )
    assert md is not None
    return md


def stamped_subject(md: dict[str, Any]) -> SubjectRef:
    return SubjectRef(str(md[K_SUBJECT]))


def stamped_scope(md: dict[str, Any]) -> Scope:
    return Scope(str(md[K_SCOPE]))


def stamped_mentions(md: dict[str, Any]) -> list[SubjectRef]:
    raw = md.get(K_MENTIONS)
    if not raw:
        return []
    return [SubjectRef(h) for h in str(raw).split(",") if h]
