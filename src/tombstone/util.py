"""Small, dependency-free helpers shared by every layer: ULIDs, canonical JSON, hashing."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from collections.abc import Mapping, Sequence
from typing import Any

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _b32(data: bytes, length: int) -> str:
    value = int.from_bytes(data, "big")
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(out))


def new_ulid(timestamp_ms: int | None = None) -> str:
    """Return a 26-character ULID (48-bit ms timestamp + 80 random bits), Crockford base32."""
    ts = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
    rand = secrets.token_bytes(10)
    return _b32(ts.to_bytes(6, "big") + rand, 26)


def derived_ulid(*parts: str) -> str:
    """A ULID-shaped identifier derived deterministically from ``parts``.

    Used where Hard Rule 9 requires the same input to produce the same id forever (trace ids).
    It is not time-ordered; it is a 128-bit truncation of SHA-256 over the parts.
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()[:16]
    return _b32(digest, 26)


def is_ulid(value: str) -> bool:
    return len(value) == 26 and all(c in _CROCKFORD for c in value)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8, floats as repr."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def content_hash(text: str) -> str:
    """SHA-256 of content. The only thing about content that is ever stored."""
    return sha256_hex(text)


def stable_int_hash(value: str, modulus: int) -> int:
    """Deterministic bucket assignment (used for subject → shard). Never Python's hash()."""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % modulus


def fingerprint_f32(vector: Sequence[float], dims: int = 32) -> str:
    """Hex of the first ``dims`` little-endian float32 values of ``vector``.

    This is the byte pattern a physical probe looks for in storage. Never the whole vector.
    """
    import struct

    head = [float(x) for x in vector[:dims]]
    if len(head) < dims:
        raise ValueError(f"embedding has {len(head)} dims; fingerprint needs at least {dims}")
    return struct.pack(f"<{dims}f", *head).hex()


def fingerprint_bytes(fingerprint_hex: str) -> bytes:
    return bytes.fromhex(fingerprint_hex)


def utc_ms() -> int:
    return int(time.time() * 1000)


def secure_write(path: str | os.PathLike[str], data: bytes, mode: int = 0o600) -> None:
    """Write ``data`` to ``path`` atomically with the given mode (used for pepper and keys)."""
    path = os.fspath(path)
    tmp = f"{path}.tmp.{secrets.token_hex(4)}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        os.chmod(path, mode)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    """Write ``text`` to ``path`` atomically: temp file, fsync, rename.

    Durable state that a reader may open at any moment (a store's tombstone set, an adapter's
    erasure state) must never be observed truncated. A plain ``write_text`` truncates first, so a
    concurrent reader sees an empty file and a crash mid-write destroys the old contents — and for
    a suppression marker, losing it silently un-suppresses erased records.
    """
    path = os.fspath(path)
    tmp = f"{path}.tmp.{secrets.token_hex(4)}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def freeze_mapping(m: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(m) if m else {}
