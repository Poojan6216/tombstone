"""Physical probes: is the artifact's byte pattern still in the store's persisted files?

Patterns:
  * the embedding fingerprint — the exact little-endian float32 sequence of the first 32 dims
    (128 bytes); a store that persists float32 holds this contiguous run;
  * the artifact id — every stamped record carries its own ``tombstone.embed_id`` /
    ``artifact_id`` in metadata, so the id string is a marker for the metadata bytes;
  * store-specific encodings (qdrant local pickles float64 big-endian; derived from the same
    float32 values, exactly).

Quantised indexes are reported ``UNVERIFIED(quantised)`` by the adapter unless the quantiser is
known. This module never decides an outcome; it reports matches and where.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Mapping
from pathlib import Path

from tombstone.stores.base import PhysicalProbeResult
from tombstone.util import fingerprint_bytes

CHUNK = 8 * 1024 * 1024


def fingerprint_patterns(fingerprint_hex: str) -> dict[str, bytes]:
    """The byte encodings a physical scan should look for, from the float32 LE fingerprint."""
    f32 = fingerprint_bytes(fingerprint_hex)
    n = len(f32) // 4
    values = struct.unpack(f"<{n}f", f32)
    return {
        "f32le": f32,
        "f32be": struct.pack(f">{n}f", *values),
        "f64le": struct.pack(f"<{n}d", *values),
        "f64be": struct.pack(f">{n}d", *values),
    }


def scan_file(path: Path, patterns: Mapping[str, bytes]) -> dict[str, int]:
    """Count occurrences of each pattern in ``path`` (streamed, overlap-safe)."""
    counts: dict[str, int] = dict.fromkeys(patterns, 0)
    if not path.is_file():
        return counts
    longest = max((len(p) for p in patterns.values()), default=0)
    tail = b""
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            buf = tail + block
            for name, pat in patterns.items():
                start = 0
                while True:
                    i = buf.find(pat, start)
                    if i < 0:
                        break
                    counts[name] += 1
                    start = i + 1
            tail = buf[-(longest - 1) :] if longest > 1 else b""
    return counts


def scan_files_for_patterns(
    paths: Iterable[Path], patterns: Mapping[str, bytes], method: str = "byte-scan"
) -> PhysicalProbeResult:
    locations: list[str] = []
    total = 0
    scanned = 0
    bytes_scanned = 0
    per_pattern: dict[str, int] = dict.fromkeys(patterns, 0)
    for p in paths:
        if not p.is_file():
            continue
        scanned += 1
        bytes_scanned += p.stat().st_size
        counts = scan_file(p, patterns)
        hit = sum(counts.values())
        for k, v in counts.items():
            per_pattern[k] += v
        if hit:
            locations.append(f"{p.name}:{'+'.join(k for k, v in counts.items() if v)}")
            total += hit
    measurement: dict[str, float] = {
        "files_scanned": float(scanned),
        "bytes_scanned": float(bytes_scanned),
        "matches": float(total),
    }
    for k, v in per_pattern.items():
        measurement[f"matches_{k}"] = float(v)
    return PhysicalProbeResult(
        found=total > 0,
        method=method,
        locations=tuple(locations),
        measurement=measurement,
        detail=(
            f"{total} match(es) in {len(locations)} file(s)"
            if total
            else f"no match in {scanned} file(s), {bytes_scanned} bytes"
        ),
    )


def walk_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())
