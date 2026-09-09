"""Capture hooks: the only place lineage nodes and edges are created for stores.

Ordering decision: lineage is written *before* the store write. A store write that then fails
leaves a phantom node (verification finds nothing, harmless); the reverse would leave
unlineaged data, which is the failure mode this tool exists to remove.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from tombstone.lineage.stamp import (
    K_ARTIFACT,
    K_CHUNK,
    K_DERIVED_FROM,
    K_EMBED,
    require_stamped,
    stamped_mentions,
    stamped_scope,
    stamped_subject,
)
from tombstone.lineage.store import LineageStore
from tombstone.model.artifacts import ArtifactKind, Scope, SubjectRef
from tombstone.model.lineage import Edge, Node
from tombstone.util import content_hash, derived_ulid, fingerprint_f32, new_ulid

CHUNK_STORE = "chunks"  # virtual store name for CHUNK nodes when no docstore holds them
SOURCE_STORE = "source"  # virtual store name for SOURCE nodes when no docstore holds them


@dataclass(frozen=True, slots=True)
class EmbedRecord:
    """What a vector store adapter needs to write one record, plus the lineage it produced."""

    key: str
    vector: list[float]
    metadata: dict[str, Any]
    document: str | None
    embed_node: Node
    chunk_node: Node


class Capture:
    def __init__(
        self,
        lineage: LineageStore,
        scope: Scope,
        source_store: str = SOURCE_STORE,
        chunk_store: str = CHUNK_STORE,
    ) -> None:
        self.lineage = lineage
        self.scope = scope
        self.source_store = source_store
        self.chunk_store = chunk_store

    # --- SOURCE ------------------------------------------------------------------------------

    def ensure_source(self, md: dict[str, Any], text: str | None = None) -> Node:
        """Create (or fetch) the SOURCE node a stamped document refers to."""
        md = require_stamped(md, "source document")
        scope = stamped_scope(md)
        if scope != self.scope:
            raise ValueError(
                f"document stamped for scope {scope.tenant!r} but capture is for {self.scope.tenant!r}"
            )
        artifact_id = str(md[K_ARTIFACT])
        existing = self.lineage.node(artifact_id)
        if existing is not None:
            return existing
        node = Node(
            artifact_id=artifact_id,
            kind=ArtifactKind.SOURCE,
            store=self.source_store,
            store_key=artifact_id,  # a docstore keys rows by artifact id; the source hash is in md
            scope=scope,
            content_hash=content_hash(text) if text is not None else "",
            embedding_fingerprint=None,
            subject_hmac=stamped_subject(md).hmac,
            created_seq=self.lineage.next_seq(),
        )
        with self.lineage.tx():
            self.lineage.add_node(node)
            for m in stamped_mentions(md):
                self.lineage.add_mention(node.artifact_id, m, scope)
            derived = md.get(K_DERIVED_FROM)
            if derived:
                self.lineage.add_edge(Edge(str(derived), node.artifact_id, "derived_from"))
        return node

    # --- CHUNK -------------------------------------------------------------------------------

    def ensure_chunk(self, md: dict[str, Any], text: str) -> tuple[Node, dict[str, Any]]:
        """Create (or fetch) the CHUNK node for a piece of text under a stamped source.

        The chunk id is derived from (source id, content hash), so the same chunk added to two
        indexes maps to one CHUNK node with two EMBED children. Returns the node and metadata
        enriched with ``tombstone.chunk_id``.
        """
        source = self.ensure_source(md)
        ch = content_hash(text)
        chunk_id = str(md.get(K_CHUNK) or derived_ulid("chunk", source.artifact_id, ch))
        node = self.lineage.node(chunk_id)
        if node is None:
            node = Node(
                artifact_id=chunk_id,
                kind=ArtifactKind.CHUNK,
                store=self.chunk_store,
                store_key=chunk_id,
                scope=source.scope,
                content_hash=ch,
                embedding_fingerprint=None,
                subject_hmac=source.subject_hmac,
                created_seq=self.lineage.next_seq(),
            )
            with self.lineage.tx():
                self.lineage.add_node(node)
                self.lineage.add_edge(Edge(source.artifact_id, chunk_id, "chunk"))
        out = dict(md)
        out[K_CHUNK] = chunk_id
        return node, out

    # --- EMBED -------------------------------------------------------------------------------

    def prepare_embeds(
        self,
        store_name: str,
        model_name: str,
        keys: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
        documents: Sequence[str | None] | None = None,
        probe_embed: Callable[[Sequence[str]], list[list[float]]] | None = None,
    ) -> list[EmbedRecord]:
        """Create CHUNK (if needed) and EMBED nodes plus ``chunk → embed`` edges; return records
        carrying the enriched metadata the adapter must store."""
        if not (len(keys) == len(vectors) == len(metadatas)):
            raise ValueError("keys, vectors and metadatas must have equal length")
        docs: Sequence[str | None] = documents if documents is not None else [None] * len(keys)
        records: list[EmbedRecord] = []
        pending_probes: list[tuple[str, str]] = []
        with self.lineage.tx():
            for key, vec, md, doc in zip(keys, vectors, metadatas, docs, strict=True):
                text = doc if doc is not None else ""
                chunk, md2 = self.ensure_chunk(md, text)
                fp = fingerprint_f32(vec)
                existing = self.lineage.node_by_store_key(store_name, key)
                if existing is not None and existing.kind is ArtifactKind.EMBED:
                    embed = existing
                else:
                    embed = Node(
                        artifact_id=new_ulid(),
                        kind=ArtifactKind.EMBED,
                        store=store_name,
                        store_key=key,
                        scope=chunk.scope,
                        content_hash=chunk.content_hash,
                        embedding_fingerprint=fp,
                        subject_hmac=chunk.subject_hmac,
                        created_seq=self.lineage.next_seq(),
                    )
                    self.lineage.add_node(embed)
                    self.lineage.add_edge(
                        Edge(chunk.artifact_id, embed.artifact_id, f"embed:{model_name}")
                    )
                    if probe_embed is not None and doc:
                        pending_probes.append((embed.artifact_id, doc))
                md3 = dict(md2)
                md3[K_EMBED] = embed.artifact_id
                records.append(
                    EmbedRecord(
                        key=key,
                        vector=[float(x) for x in vec],
                        metadata=md3,
                        document=doc,
                        embed_node=embed,
                        chunk_node=chunk,
                    )
                )
        if pending_probes and probe_embed is not None:
            # one embedding call for every probe query in the batch, not three per chunk
            from tombstone.verify.logical import record_probes_batch

            record_probes_batch(self.lineage, model_name, pending_probes, probe_embed)
        return records

    # --- CACHE -------------------------------------------------------------------------------

    def record_cache(
        self,
        store_name: str,
        key: str,
        parent_chunk_ids: Sequence[str],
        answer_hash: str,
        subject: SubjectRef | None,
        fingerprint: str | None = None,
        via: str = "cache:exact",
    ) -> list[Node]:
        """One CACHE node per (store, key) with an edge from every parent chunk.

        A cache entry that descends from chunks of several subjects gets one node per subject
        so each subject's trace reaches it (it is erased when any of them is erased).
        """
        parents = [p for p in self.lineage.nodes(list(parent_chunk_ids)) if p is not None]
        subjects = sorted({p.subject_hmac for p in parents}) or ([subject.hmac] if subject else [])
        nodes: list[Node] = []
        with self.lineage.tx():
            for s in subjects:
                existing = self.lineage.node_by_store_key(store_name, f"{key}@{s[:16]}")
                if existing is not None:
                    nodes.append(existing)
                    continue
                node = Node(
                    artifact_id=new_ulid(),
                    kind=ArtifactKind.CACHE,
                    store=store_name,
                    store_key=f"{key}@{s[:16]}",
                    scope=self.scope,
                    content_hash=answer_hash,
                    embedding_fingerprint=fingerprint,
                    subject_hmac=s,
                    created_seq=self.lineage.next_seq(),
                )
                self.lineage.add_node(node)
                for p in parents:
                    if p.subject_hmac == s:
                        self.lineage.add_edge(Edge(p.artifact_id, node.artifact_id, via))
                nodes.append(node)
        return nodes

    # --- native delete ----------------------------------------------------------------------------

    def record_native_delete(self, store_name: str, keys: Sequence[str]) -> list[str]:
        """The app called the store's own ``delete()``. Mark the EMBED nodes so ``verify`` can
        say 'logical PASS, physical FAIL' about them. Returns the artifact ids marked."""
        ids: list[str] = []
        for k in keys:
            n = self.lineage.node_by_store_key(store_name, k)
            if n is not None:
                ids.append(n.artifact_id)
        if ids:
            self.lineage.tombstone(ids, reason="native-delete", trace_id=None)
        return ids

    def register(self, store_name: str, kind: str) -> None:
        self.lineage.register_store(store_name, kind, self.scope)
