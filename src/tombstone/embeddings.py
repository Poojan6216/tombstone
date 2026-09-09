"""Embedders: the store input. Not a decision-maker (Hard Rule 1).

``HashEmbedder`` is deterministic and dependency-free (tests). ``SentenceTransformerEmbedder``
wraps sentence-transformers (bench). ``LangChainEmbedder`` adapts any LangChain ``Embeddings``.
Every embedder rounds to float32 so the stored bytes equal the fingerprint bytes.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    name: str
    dims: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def to_f32(vector: Sequence[float]) -> list[float]:
    """Round-trip through little-endian float32 so the value equals what the store will hold."""
    packed = struct.pack(f"<{len(vector)}f", *[float(x) for x in vector])
    return list(struct.unpack(f"<{len(vector)}f", packed))


class HashEmbedder:
    """Deterministic pseudo-embedding: a unit vector from SHA-256 of word shingles.

    Similar texts (shared shingles) get similar vectors, which is enough for probe tests to
    have power without downloading a model. Never use for real retrieval quality.
    """

    def __init__(self, dims: int = 64, name: str = "hash-embed") -> None:
        if dims < 32:
            raise ValueError("physical probes need at least 32 dims")
        self.dims = dims
        self.name = f"{name}-{dims}"

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        tokens = text.lower().split()
        grams = tokens + [" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
        if not grams:
            grams = [text]
        for g in grams:
            h = hashlib.sha256(g.encode("utf-8")).digest()
            for j in range(0, 32, 2):
                idx = int.from_bytes(h[j : j + 2], "big") % self.dims
                # sign from a byte that does not feed the index (index parity would otherwise
                # equal the sign bit and every vector would share one sign pattern)
                sign = 1.0 if h[31 - j // 2] & 0x80 else -1.0
                vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return to_f32([v / norm for v in vec])

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self.name = model_name.rsplit("/", maxsplit=1)[-1]
        self.dims = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        arr = self._model.encode(
            list(texts), batch_size=64, normalize_embeddings=True, convert_to_numpy=True
        )
        return [to_f32([float(x) for x in row]) for row in arr]


class LangChainEmbedder:
    def __init__(self, embeddings: Any, name: str, dims: int | None = None) -> None:
        self._e = embeddings
        self.name = name
        self.dims = dims if dims is not None else len(to_f32(embeddings.embed_query("probe")))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [to_f32(v) for v in self._e.embed_documents(list(texts))]


def get_embedder(name: str, dims: int = 64) -> Embedder:
    """``hash-embed`` → HashEmbedder; anything else → sentence-transformers by model name."""
    if name.startswith("hash-embed"):
        try:
            d = int(name.rsplit("-", 1)[1])
        except (ValueError, IndexError):
            d = dims
        return HashEmbedder(d)
    aliases = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
        "gtr-t5-base": "sentence-transformers/gtr-t5-base",
    }
    return SentenceTransformerEmbedder(aliases.get(name, name))
