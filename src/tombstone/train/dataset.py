"""Fine-tune dataset builder with lineage (torch-free).

Examples are sharded by a stable hash of ``subject_hmac`` (default 16 shards) so every example of
one subject lands in one shard — the precondition for exact unlearning by shard retrain.
``manifest.json`` holds ids, content hashes, shard numbers and parent chunk ids; never content.
Content lives in ``shard-NN.jsonl`` beside it (the app's training data, not ``.tombstone/``).

``DatasetStore`` is the ``ErasableStore`` view: suppress marks an example (the trainer skips it),
reclaim drops the row, rewrites the shard file, re-hashes the manifest and bumps the pin.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tombstone.lineage.capture import Capture
from tombstone.model.artifacts import ArtifactKind, ArtifactRef
from tombstone.model.lineage import Edge, Node
from tombstone.model.pins import ManifestPin
from tombstone.model.status import VerifyLevel
from tombstone.stores.base import (
    LogicalProbeResult,
    PhysicalProbeResult,
    ProbeSet,
    ReclaimResult,
)
from tombstone.util import (
    atomic_write_text,
    canonical_json,
    content_hash,
    derived_ulid,
    sha256_hex,
    stable_int_hash,
)

MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    text: str
    chunk_id: str
    subject_hmac: str
    shard: int


def shard_for(subject_hmac: str, shards: int) -> int:
    return stable_int_hash(subject_hmac, shards)


def manifest_hash(manifest: dict[str, Any]) -> str:
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    return sha256_hex(canonical_json(body))


def build_dataset(
    capture: Capture,
    store_name: str,
    manifest_path: str | Path,
    chunks: Sequence[tuple[Node, str]],
    shards: int = 16,
) -> dict[str, Any]:
    """Write shard files + manifest and create TRAIN nodes with ``chunk → train`` edges.

    ``chunks`` are (CHUNK node, text) pairs from the lineage store / docstore.
    """
    manifest_path = Path(manifest_path)
    out_dir = manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    per_shard: dict[int, list[dict[str, Any]]] = {i: [] for i in range(shards)}
    entries: list[dict[str, Any]] = []
    lineage = capture.lineage
    capture.register(store_name, "dataset")
    with lineage.tx():
        for chunk, text in chunks:
            if chunk.kind is not ArtifactKind.CHUNK:
                raise ValueError(f"{chunk.artifact_id} is a {chunk.kind.value}, not a chunk")
            shard = shard_for(chunk.subject_hmac, shards)
            example_id = derived_ulid("train", store_name, chunk.artifact_id, str(shard))
            existing = lineage.node(example_id)
            if existing is not None:
                node = existing
            else:
                node = Node(
                    artifact_id=example_id,
                    kind=ArtifactKind.TRAIN,
                    store=store_name,
                    store_key=example_id,
                    scope=chunk.scope,
                    content_hash=content_hash(text),
                    embedding_fingerprint=None,
                    subject_hmac=chunk.subject_hmac,
                    created_seq=lineage.next_seq(),
                )
                lineage.add_node(node)
                lineage.add_edge(Edge(chunk.artifact_id, node.artifact_id, f"train:shard-{shard}"))
            per_shard[shard].append({"id": node.artifact_id, "text": text})
            entries.append(
                {
                    "id": node.artifact_id,
                    "content_hash": node.content_hash,
                    "shard": shard,
                    "parents": [chunk.artifact_id],
                    "subject": chunk.subject_hmac,
                    "suppressed": False,
                }
            )
    for shard, rows in per_shard.items():
        _write_jsonl(out_dir / f"shard-{shard:02d}.jsonl", rows)
    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "store": store_name,
        "shards": shards,
        "examples": sorted(entries, key=lambda e: e["id"]),
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    atomic_write_text(manifest_path, json.dumps(manifest, indent=1, sort_keys=True))
    lineage.put_pin(
        ManifestPin(store_name, manifest["manifest_hash"], len(entries)), "dataset build"
    )
    return manifest


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


class DatasetStore:
    kind = "dataset"

    def __init__(self, name: str, manifest_path: str | Path) -> None:
        self.name = name
        self.manifest_path = Path(manifest_path)
        self.dir = self.manifest_path.parent
        self.capabilities: frozenset[VerifyLevel] = frozenset(
            {VerifyLevel.LOGICAL, VerifyLevel.PHYSICAL}
        )

    def version(self) -> str:
        return f"manifest v{MANIFEST_VERSION}"

    def close(self) -> None:
        return None

    # --- manifest ----------------------------------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"version": MANIFEST_VERSION, "store": self.name, "shards": 0, "examples": []}
        m: dict[str, Any] = dict(json.loads(self.manifest_path.read_text(encoding="utf-8")))
        if manifest_hash(m) != m.get("manifest_hash"):
            from tombstone.errors import PinMismatch

            raise PinMismatch(
                f"dataset manifest {self.manifest_path} hash does not match its contents; it was "
                "edited outside tombstone. Rebuild the dataset or run `tombstone repin --reason`."
            )
        return m

    def _write_manifest(self, m: dict[str, Any]) -> str:
        m["manifest_hash"] = manifest_hash(m)
        # Atomic: suppress and reclaim rewrite this while other sagas read it. A truncating write
        # let a reader see an empty file (JSONDecodeError mid-erasure under twenty concurrent
        # sagas) or, worse, a partial one that still parsed — which fails the manifest_hash check
        # and reports PinMismatch, accusing the operator of editing the manifest by hand.
        atomic_write_text(self.manifest_path, json.dumps(m, indent=1, sort_keys=True))
        return str(m["manifest_hash"])

    def pin(self) -> ManifestPin:
        m = self.manifest()
        return ManifestPin(self.name, str(m.get("manifest_hash", "")), len(m["examples"]))

    def shard_examples(self, shard: int, include_suppressed: bool = False) -> list[Example]:
        m = self.manifest()
        suppressed = {e["id"] for e in m["examples"] if e.get("suppressed")}
        rows = read_jsonl(self.dir / f"shard-{shard:02d}.jsonl")
        by_id = {e["id"]: e for e in m["examples"]}
        out: list[Example] = []
        for r in rows:
            if r["id"] in suppressed and not include_suppressed:
                continue
            e = by_id.get(r["id"])
            if e is None:
                continue
            out.append(Example(r["id"], r["text"], e["parents"][0], e["subject"], int(e["shard"])))
        return out

    def shards(self) -> int:
        return int(self.manifest().get("shards", 0))

    # --- ErasableStore -----------------------------------------------------------------------

    @staticmethod
    def _example_id(ref: ArtifactRef) -> str:
        return ref.artifact_id

    def suppress(self, refs: Sequence[ArtifactRef]) -> None:
        m = self.manifest()
        ids = {self._example_id(r) for r in refs}
        for e in m["examples"]:
            if e["id"] in ids:
                e["suppressed"] = True
        self._write_manifest(m)

    def reclaim(self, refs: Sequence[ArtifactRef]) -> ReclaimResult:
        m = self.manifest()
        ids = {self._example_id(r) for r in refs}
        before = len(m["examples"])
        touched_shards = {int(e["shard"]) for e in m["examples"] if e["id"] in ids}
        m["examples"] = [e for e in m["examples"] if e["id"] not in ids]
        for shard in touched_shards:
            path = self.dir / f"shard-{shard:02d}.jsonl"
            rows = [r for r in read_jsonl(path) if r["id"] not in ids]
            _write_jsonl(path, rows)
        new_hash = self._write_manifest(m)
        deleted = before - len(m["examples"])
        return ReclaimResult(
            noop=deleted == 0,
            method="drop row, re-hash manifest",
            measurement={"deleted": float(deleted), "shards_rewritten": float(len(touched_shards))},
            detail=f"manifest {new_hash[:12]}",
        )

    def probe_logical(self, ref: ArtifactRef, probes: ProbeSet) -> LogicalProbeResult:
        m = self.manifest()
        eid = self._example_id(ref)
        entry = next((e for e in m["examples"] if e["id"] == eid), None)
        found = entry is not None and not entry.get("suppressed")
        return LogicalProbeResult(found=found, found_by=("id",) if found else (), probes_run=1)

    def probe_physical(self, ref: ArtifactRef) -> PhysicalProbeResult:
        """Re-hash every row of every shard file: the artifact's id and content hash must be
        absent. (We never hold the content, so hashing rows is the strongest check.) Rows of
        other examples with identical text also match; the saga compares against a baseline."""
        eid = self._example_id(ref)
        locations: list[str] = []
        rows_scanned = 0
        id_matches = 0
        hash_matches = 0
        for path in sorted(self.dir.glob("shard-*.jsonl")):
            for r in read_jsonl(path):
                rows_scanned += 1
                is_id = r.get("id") == eid
                is_hash = content_hash(str(r.get("text", ""))) == ref.content_hash
                if is_id:
                    id_matches += 1
                if is_hash:  # our own row counts here too, so a baseline includes our copy
                    hash_matches += 1
                if (is_id or is_hash) and path.name not in locations:
                    locations.append(path.name)
        matches = max(id_matches, hash_matches)
        return PhysicalProbeResult(
            found=matches > 0,
            method="hash-scan of shard rows",
            locations=tuple(locations),
            measurement={
                "rows_scanned": float(rows_scanned),
                "matches": float(matches),
                "matches_id": float(id_matches),
                "matches_hash": float(hash_matches),
            },
        )

    def live_content_duplicates(self, ref: ArtifactRef) -> int:
        eid = self._example_id(ref)
        return sum(
            1
            for e in self.manifest()["examples"]
            if e["id"] != eid and e["content_hash"] == ref.content_hash and not e.get("suppressed")
        )

    def count(self) -> int:
        return len(self.manifest()["examples"])

    def sample_keys(self, n: int) -> list[str]:
        return [e["id"] for e in self.manifest()["examples"][:n]]
