"""Core value types. Frozen dataclasses, no I/O, no dependencies beyond the stdlib."""

from __future__ import annotations

from tombstone.model.artifacts import ArtifactKind, ArtifactRef, Scope, SubjectRef
from tombstone.model.lineage import Edge, LineageSnapshot, Node, Trace
from tombstone.model.pins import ManifestPin, ModelPin, StorePin
from tombstone.model.status import ArtifactStatus, Outcome, Receipt, VerifyLevel

__all__ = [
    "ArtifactKind",
    "ArtifactRef",
    "ArtifactStatus",
    "Edge",
    "LineageSnapshot",
    "ManifestPin",
    "ModelPin",
    "Node",
    "Outcome",
    "Receipt",
    "Scope",
    "StorePin",
    "SubjectRef",
    "Trace",
    "VerifyLevel",
]
