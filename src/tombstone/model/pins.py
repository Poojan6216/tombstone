"""Pins: what the tool believed about a store, a model, or a manifest when it last looked.

A silent change (backend upgraded, capabilities lost, adapter retrained, manifest edited) must be
acknowledged with ``tombstone repin --reason`` before an erasure is allowed to proceed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tombstone.model.status import VerifyLevel


@dataclass(frozen=True, slots=True)
class PinDelta:
    pin_name: str
    field: str
    before: str
    after: str

    def __str__(self) -> str:
        return f"{self.pin_name}: {self.field} changed {self.before!r} → {self.after!r}"


@dataclass(frozen=True, slots=True)
class StorePin:
    name: str
    backend: str  # "chroma", "pgvector", ...
    version: str  # backend/library version string
    capabilities: frozenset[VerifyLevel]
    embedding_model: str = ""
    dims: int = 0

    kind: str = "store"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "backend": self.backend,
            "version": self.version,
            "capabilities": sorted(c.value for c in self.capabilities),
            "embedding_model": self.embedding_model,
            "dims": self.dims,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> StorePin:
        return StorePin(
            name=str(d["name"]),
            backend=str(d["backend"]),
            version=str(d["version"]),
            capabilities=frozenset(VerifyLevel(c) for c in d.get("capabilities", [])),
            embedding_model=str(d.get("embedding_model", "")),
            dims=int(d.get("dims", 0)),
        )

    def diff(self, other: StorePin) -> list[PinDelta]:
        out: list[PinDelta] = []
        if self.backend != other.backend:
            out.append(PinDelta(self.name, "backend", self.backend, other.backend))
        if self.version != other.version:
            out.append(PinDelta(self.name, "version", self.version, other.version))
        if self.capabilities != other.capabilities:
            out.append(
                PinDelta(
                    self.name,
                    "capabilities",
                    ",".join(sorted(c.value for c in self.capabilities)) or "∅",
                    ",".join(sorted(c.value for c in other.capabilities)) or "∅",
                )
            )
        if self.embedding_model != other.embedding_model:
            out.append(
                PinDelta(self.name, "embedding_model", self.embedding_model, other.embedding_model)
            )
        if self.dims != other.dims:
            out.append(PinDelta(self.name, "dims", str(self.dims), str(other.dims)))
        return out


@dataclass(frozen=True, slots=True)
class ModelPin:
    name: str
    base_model: str
    adapter_config_hash: str  # hash over adapter_config.json + weights per shard
    shard_count: int

    kind: str = "model"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "base_model": self.base_model,
            "adapter_config_hash": self.adapter_config_hash,
            "shard_count": self.shard_count,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ModelPin:
        return ModelPin(
            name=str(d["name"]),
            base_model=str(d["base_model"]),
            adapter_config_hash=str(d["adapter_config_hash"]),
            shard_count=int(d["shard_count"]),
        )

    def diff(self, other: ModelPin) -> list[PinDelta]:
        out: list[PinDelta] = []
        if self.base_model != other.base_model:
            out.append(PinDelta(self.name, "base_model", self.base_model, other.base_model))
        if self.adapter_config_hash != other.adapter_config_hash:
            out.append(
                PinDelta(
                    self.name,
                    "adapter_config_hash",
                    self.adapter_config_hash[:12],
                    other.adapter_config_hash[:12],
                )
            )
        if self.shard_count != other.shard_count:
            out.append(
                PinDelta(self.name, "shard_count", str(self.shard_count), str(other.shard_count))
            )
        return out


@dataclass(frozen=True, slots=True)
class ManifestPin:
    name: str
    manifest_hash: str
    example_count: int

    kind: str = "manifest"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "manifest_hash": self.manifest_hash,
            "example_count": self.example_count,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ManifestPin:
        return ManifestPin(
            name=str(d["name"]),
            manifest_hash=str(d["manifest_hash"]),
            example_count=int(d["example_count"]),
        )

    def diff(self, other: ManifestPin) -> list[PinDelta]:
        out: list[PinDelta] = []
        if self.manifest_hash != other.manifest_hash:
            out.append(
                PinDelta(
                    self.name, "manifest_hash", self.manifest_hash[:12], other.manifest_hash[:12]
                )
            )
        if self.example_count != other.example_count:
            out.append(
                PinDelta(
                    self.name, "example_count", str(self.example_count), str(other.example_count)
                )
            )
        return out


Pin = StorePin | ModelPin | ManifestPin


def pin_from_dict(d: dict[str, Any]) -> Pin:
    kind = d.get("kind")
    if kind == "store":
        return StorePin.from_dict(d)
    if kind == "model":
        return ModelPin.from_dict(d)
    if kind == "manifest":
        return ManifestPin.from_dict(d)
    raise ValueError(f"unknown pin kind {kind!r}")


def diff_pins(before: Pin, after: Pin) -> list[PinDelta]:
    if type(before) is not type(after):
        return [PinDelta(before.name, "kind", before.kind, after.kind)]
    if isinstance(before, StorePin) and isinstance(after, StorePin):
        return before.diff(after)
    if isinstance(before, ModelPin) and isinstance(after, ModelPin):
        return before.diff(after)
    if isinstance(before, ManifestPin) and isinstance(after, ManifestPin):
        return before.diff(after)
    raise TypeError("unreachable")
