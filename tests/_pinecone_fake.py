"""An in-process stand-in for the Pinecone client, faithful to the parts the adapter depends on.

This is not a claim that the real service behaves this way in every respect — a fake written by
the same person as the adapter can only test the adapter against its author's beliefs, never
against Pinecone. It exists to exercise the behaviours the adapter is *built around*, which no
happy-path mock would catch:

* every method is keyword-only, as the real client's are, so a positional call fails here too;
* writes are **eventually consistent**: a delete or a metadata update is not visible until a
  configurable number of subsequent reads have gone by. This is the behaviour the adapter's
  settle loop exists for, and with ``lag=0`` the loop would never be tested at all;
* ``fetch`` returns an object with a ``.vectors`` dict, ``query`` an object with ``.matches``,
  ``describe_index_stats`` one with ``.namespaces`` and ``.total_vector_count`` — the shapes the
  adapter reads attributes off.

Anything the adapter does not use is deliberately absent, so a future change that reaches for
another API fails loudly here instead of silently passing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Vec:
    id: str
    values: list[float]
    metadata: dict[str, Any]


@dataclass
class _Match:
    id: str
    score: float
    metadata: dict[str, Any]


@dataclass
class _FetchResponse:
    vectors: dict[str, _Vec]


@dataclass
class _QueryResponse:
    matches: list[_Match]


@dataclass
class _NamespaceStats:
    vector_count: int


@dataclass
class _StatsResponse:
    namespaces: dict[str, _NamespaceStats]
    total_vector_count: int


@dataclass
class _Pending:
    """A write the service has accepted but not yet made visible."""

    apply: Any
    reads_left: int


@dataclass
class FakeIndex:
    lag: int = 2
    data: dict[str, dict[str, _Vec]] = field(default_factory=dict)
    pending: list[_Pending] = field(default_factory=list)
    closed: bool = False
    calls: list[str] = field(default_factory=list)

    # --- consistency ---------------------------------------------------------------------

    def _tick(self) -> None:
        """A read advances the clock; writes older than ``lag`` reads become visible."""
        still: list[_Pending] = []
        for p in self.pending:
            p.reads_left -= 1
            if p.reads_left <= 0:
                p.apply()
            else:
                still.append(p)
        self.pending = still

    def _later(self, fn: Any) -> None:
        if self.lag <= 0:
            fn()
        else:
            self.pending.append(_Pending(fn, self.lag))

    def _ns(self, namespace: str) -> dict[str, _Vec]:
        return self.data.setdefault(namespace, {})

    # --- the API the adapter uses -------------------------------------------------------

    def upsert(self, *, vectors: Any, namespace: str = "", **_: Any) -> dict[str, int]:
        self.calls.append("upsert")
        ns = self._ns(namespace)
        for v in vectors:
            ns[str(v["id"])] = _Vec(str(v["id"]), list(v["values"]), dict(v.get("metadata") or {}))
        return {"upserted_count": len(list(vectors))}

    def fetch(self, *, ids: Any, namespace: str = "", **_: Any) -> _FetchResponse:
        self.calls.append("fetch")
        self._tick()
        ns = self._ns(namespace)
        return _FetchResponse(vectors={i: ns[i] for i in ids if i in ns})

    def query(
        self,
        *,
        top_k: int,
        vector: Any = None,
        namespace: str = "",
        filter: Any = None,  # noqa: A002 - the real client names it this
        include_metadata: bool = False,
        **_: Any,
    ) -> _QueryResponse:
        self.calls.append("query")
        self._tick()
        rows = list(self._ns(namespace).values())
        if filter:
            for key, cond in filter.items():
                want = cond["$eq"] if isinstance(cond, dict) and "$eq" in cond else cond
                rows = [r for r in rows if r.metadata.get(key) == want]
        if vector is not None and any(vector):
            rows.sort(key=lambda r: -_cosine(r.values, vector))
        return _QueryResponse(
            matches=[
                _Match(r.id, _cosine(r.values, vector) if vector else 0.0, dict(r.metadata))
                for r in rows[:top_k]
            ]
        )

    def delete(self, *, ids: Any = None, namespace: str = "", **_: Any) -> dict[str, Any]:
        self.calls.append("delete")
        ns = self._ns(namespace)
        targets = list(ids or [])

        def apply() -> None:
            for i in targets:
                ns.pop(i, None)

        self._later(apply)
        return {}

    def update(
        self,
        *,
        id: str,
        set_metadata: Any = None,
        namespace: str = "",
        **_: Any,  # noqa: A002
    ) -> dict[str, Any]:
        self.calls.append("update")
        ns = self._ns(namespace)
        if id not in ns:
            raise KeyError(id)  # the real client errors on an unknown id
        patch = dict(set_metadata or {})

        def apply() -> None:
            if id in ns:
                ns[id].metadata.update(patch)

        self._later(apply)
        return {}

    def describe_index_stats(self, **_: Any) -> _StatsResponse:
        self.calls.append("describe_index_stats")
        self._tick()
        spaces = {k: _NamespaceStats(len(v)) for k, v in self.data.items()}
        return _StatsResponse(spaces, sum(len(v) for v in self.data.values()))

    def list(self, *, namespace: str = "", **_: Any) -> Any:
        self.calls.append("list")
        self._tick()
        yield list(self._ns(namespace))

    def close(self) -> None:
        self.closed = True


def _cosine(a: Any, b: Any) -> float:
    if not a or not b:
        return 0.0
    num = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


@dataclass
class FakePinecone:
    api_key: str
    index: FakeIndex

    def Index(self, name: str = "", host: str = "") -> FakeIndex:  # noqa: N802 - the real name
        return self.index


def install(monkeypatch: Any, lag: int = 2) -> FakeIndex:
    """Point ``pinecone.Pinecone`` at the fake for the duration of a test."""
    import pinecone

    shared = FakeIndex(lag=lag)
    monkeypatch.setattr(pinecone, "Pinecone", lambda api_key="", **_: FakePinecone(api_key, shared))
    return shared
