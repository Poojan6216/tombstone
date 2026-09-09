"""LangGraph memory adapter (Phase 8.2, optional).

``TombstoneMemoryStore`` implements the subset of LangGraph's ``BaseStore`` interface that agent
memory uses — ``put``, ``get``, ``delete``, ``search`` over ``(namespace, key)`` — on top of the
in-repo ``MemoryStore`` adapter, and captures every entry as a MEMORY node with a subject stamp
and an edge from the subject's SOURCE, so erasure runs through the same saga. It does not import
``langgraph``; when ``langgraph`` is installed, subclass ``BaseStore`` and delegate to this.

Values must be stamped: ``put(namespace, key, value, subject_id=..., source_id=...)`` stamps
them with the installation pepper. The value's content is hashed for lineage; the memory store
holds the value itself (the app's data), never ``.tombstone/``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from tombstone.lineage.capture import Capture
from tombstone.lineage.stamp import K_ARTIFACT, K_SUBJECT, stamp
from tombstone.model.artifacts import ArtifactKind
from tombstone.model.lineage import Edge, Node
from tombstone.stores.memory import MemoryStore
from tombstone.util import content_hash, derived_ulid


@dataclass(frozen=True, slots=True)
class Item:
    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any]


def _skey(namespace: Sequence[str], key: str) -> str:
    return "/".join(namespace) + "#" + key


class TombstoneMemoryStore:
    def __init__(self, store: MemoryStore, capture: Capture, pepper: bytes) -> None:
        self.store = store
        self.capture = capture
        self.pepper = pepper
        capture.register(store.name, store.kind)

    def put(
        self,
        namespace: Sequence[str],
        key: str,
        value: dict[str, Any],
        *,
        subject_id: str,
        source_id: str | None = None,
    ) -> str:
        md = stamp(
            {},
            subject_id,
            source_id or f"memory:{'/'.join(namespace)}",
            self.capture.scope.tenant,
            pepper=self.pepper,
        )
        source = self.capture.ensure_source(md)
        skey = _skey(namespace, key)
        aid = derived_ulid("memory", self.store.name, skey)
        lineage = self.capture.lineage
        text = json.dumps(value, sort_keys=True)
        node = lineage.node(aid)
        if node is None:
            node = Node(
                artifact_id=aid,
                kind=ArtifactKind.MEMORY,
                store=self.store.name,
                store_key=skey,
                scope=self.capture.scope,
                content_hash=content_hash(text),
                embedding_fingerprint=None,
                subject_hmac=str(md[K_SUBJECT]),
                created_seq=lineage.next_seq(),
            )
            with lineage.tx():
                lineage.add_node(node)
                lineage.add_edge(Edge(source.artifact_id, aid, "memory"))
        self.store.put(
            skey,
            {
                "namespace": list(namespace),
                "key": key,
                "value": value,
                K_ARTIFACT: aid,
                K_SUBJECT: md[K_SUBJECT],
            },
        )
        return aid

    def get(self, namespace: Sequence[str], key: str) -> Item | None:
        row = self.store.get(_skey(namespace, key))
        if row is None:
            return None
        return Item(tuple(row["namespace"]), row["key"], dict(row["value"]))

    def delete(self, namespace: Sequence[str], key: str) -> None:
        """The app's own delete: removes the row and records a native-delete tombstone."""
        skey = _skey(namespace, key)
        aid = derived_ulid("memory", self.store.name, skey)
        self.store._rows.pop(skey, None)
        self.capture.lineage.tombstone([aid], "native-delete", None)

    def search(
        self, namespace_prefix: Sequence[str], query: str | None = None, limit: int = 10
    ) -> list[Item]:
        prefix = "/".join(namespace_prefix)
        out: list[Item] = []
        for k in sorted(self.store._rows):
            row = self.store.get(k)
            if row is None or not k.startswith(prefix):
                continue
            if query and query.lower() not in json.dumps(row["value"]).lower():
                continue
            out.append(Item(tuple(row["namespace"]), row["key"], dict(row["value"])))
            if len(out) >= limit:
                break
        return out

    def list_namespaces(self) -> list[tuple[str, ...]]:
        out: set[tuple[str, ...]] = set()
        for k in list(self.store._rows):
            row = self.store.get(k)
            if row is not None:
                out.add(tuple(row["namespace"]))
        return sorted(out)

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        return [self.get(*op) if isinstance(op, tuple) else None for op in ops]
