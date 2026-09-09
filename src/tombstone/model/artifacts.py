"""Artifact identity: what a thing is, where it lives, and which subject it descends from."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ArtifactKind(StrEnum):
    SOURCE = "source"  # the original document / row
    CHUNK = "chunk"
    EMBED = "embed"  # one vector in one index
    CACHE = "cache"  # a cached answer or cached embedding
    TRAIN = "train"  # one training example in a dataset manifest
    ADAPTER = "adapter"  # a LoRA adapter (or shard adapter)
    MEMORY = "memory"  # LangGraph/agent memory entry (Phase 8)


@dataclass(frozen=True, slots=True)
class SubjectRef:
    """HMAC-SHA256(pepper, raw_subject_id). The raw id is never stored (Hard Rule 7)."""

    hmac: str

    @staticmethod
    def from_raw(raw_subject_id: str, pepper: bytes) -> SubjectRef:
        if not raw_subject_id:
            raise ValueError("subject id must be non-empty")
        if len(pepper) < 16:
            raise ValueError("pepper must be at least 16 bytes")
        digest = hmac.new(pepper, raw_subject_id.encode("utf-8"), hashlib.sha256).hexdigest()
        return SubjectRef(hmac=digest)

    @property
    def short(self) -> str:
        return f"hmac:{self.hmac[:4]}…{self.hmac[-4:]}"

    def __str__(self) -> str:
        return f"hmac:{self.hmac}"


@dataclass(frozen=True, slots=True)
class Scope:
    """Tenant boundary (Hard Rule 8). ``default`` if single-tenant."""

    tenant: str

    def __post_init__(self) -> None:
        if not self.tenant:
            raise ValueError("scope tenant must be non-empty")

    def __str__(self) -> str:
        return self.tenant


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str  # ULID
    kind: ArtifactKind
    store: str  # e.g. "chroma:kb-v2", "pgvector:kb-v1", "lora/support-v3"
    store_key: str  # the id the store knows it by
    scope: Scope
    content_hash: str  # SHA-256 of the content at creation. Never the content.
    embedding_fingerprint: str | None = None  # first 32 float32 dims, hex — for physical probes

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "store": self.store,
            "store_key": self.store_key,
            "scope": self.scope.tenant,
            "content_hash": self.content_hash,
            "embedding_fingerprint": self.embedding_fingerprint,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=str(d["artifact_id"]),
            kind=ArtifactKind(d["kind"]),
            store=str(d["store"]),
            store_key=str(d["store_key"]),
            scope=Scope(str(d["scope"])),
            content_hash=str(d["content_hash"]),
            embedding_fingerprint=(
                str(d["embedding_fingerprint"]) if d.get("embedding_fingerprint") else None
            ),
        )
