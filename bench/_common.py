"""Shared bench plumbing: config, corpus ingestion into every backend, embedding cache, timing."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Intel macOS: torch (libiomp5) and faiss-cpu (libomp) are two OpenMP runtimes in one process;
# multi-threaded use of either can segfault. Torch's own thread count is set separately
# (TOMBSTONE_TORCH_THREADS) after import; OpenMP pools stay at one thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

WORK = ROOT / "bench" / "_work"
RESULTS = ROOT / "bench" / "results"
PLOTS = ROOT / "bench" / "plots"
EMBED_CACHE = WORK / "embeddings"

from tombstone.embeddings import Embedder, get_embedder
from tombstone.util import sha256_hex

BACKEND_NAMES = {
    "chroma": "chroma:kb-v2",
    "faiss": "faiss:kb-v1",
    "qdrant": "qdrant:kb",
    "pgvector": "pgvector:kb-v1",
}
EMBED_MODEL = os.environ.get("TOMBSTONE_BENCH_EMBED", "all-MiniLM-L6-v2")
EMBED_MODEL_2 = os.environ.get("TOMBSTONE_BENCH_EMBED_2", "bge-small-en-v1.5")


def now_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def git_head() -> str:
    try:
        import subprocess

        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def machine() -> dict[str, Any]:
    import platform

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
    }


@dataclass
class Timer:
    t0: float = 0.0

    def __enter__(self) -> Timer:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a: object) -> None:
        self.elapsed = time.perf_counter() - self.t0


class CachedEmbedder:
    """Embeds through a content-hash cache on disk so 20 index builds pay for one model pass."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.name = inner.name
        self.dims = inner.dims
        self.path = EMBED_CACHE / f"{inner.name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self._cache[d["h"]] = d["v"]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        hashes = [sha256_hex(t) for t in texts]
        missing = [(h, t) for h, t in zip(hashes, texts, strict=True) if h not in self._cache]
        if missing:
            vecs = self.inner.embed([t for _, t in missing])
            with self.path.open("a", encoding="utf-8") as fh:
                for (h, _), v in zip(missing, vecs, strict=True):
                    self._cache[h] = v
                    fh.write(json.dumps({"h": h, "v": v}) + "\n")
        return [self._cache[h] for h in hashes]


def embedder(name: str = EMBED_MODEL) -> CachedEmbedder:
    return CachedEmbedder(get_embedder(name))


def pg_dsn() -> str | None:
    """A Postgres for the bench: TOMBSTONE_PG_DSN, else a temporary local server (tests/_pg.py)."""
    dsn = os.environ.get("TOMBSTONE_PG_DSN")
    if dsn:
        return dsn
    try:
        from tests import _pg
    except ImportError:
        return None
    handle = _pg.acquire()
    if handle is None:
        return None
    _PG_HANDLES.append(handle)
    return handle.dsn


_PG_HANDLES: list[Any] = []


def fresh_pg_database(base_dsn: str, name: str) -> str:
    import psycopg

    with psycopg.connect(base_dsn, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}"')
        conn.execute(f'CREATE DATABASE "{name}"')
    base, _, _ = base_dsn.rpartition("/")
    dsn = f"{base}/{name}"
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS pgstattuple")
        except Exception:
            pass
    return dsn


CONFIG = """\
version: 1
scope: default
lineage: {{ backend: sqlite, path: {root}/.tombstone/lineage.db }}
stores:
  - {{ name: "docs", kind: docstore, path: {root}/docs.sqlite }}
{stores}
  - {{ name: "exact-cache", kind: cache_exact, path: {root}/.langchain.db }}
  - {{ name: "semantic-cache", kind: cache_semantic, backing: "{backing}" }}
  - {{ name: "ft-dataset", kind: dataset, manifest: {root}/train/manifest.json }}
{adapter}
erase:
  require_confirm: true
  reclaim_timeout_s: 600
  semantic_probe_budget: 5
model:
  base: {base_model}
  unlearn: {unlearn}
  mia_reference: {root}/train/holdout.jsonl
out_of_scope:
  - "database backups and snapshots"
  - "write-ahead logs and replicas"
  - "embedding-provider-side request logs"
"""

STORE_LINES = {
    "chroma": '  - {{ name: "chroma:kb-v2", kind: chroma, path: {root}/chroma, collection: tombstone-kb, embedding: {embed} }}',
    "faiss": '  - {{ name: "faiss:kb-v1", kind: faiss, path: {root}/faiss/kb.index, embedding: {embed} }}',
    "qdrant": '  - {{ name: "qdrant:kb", kind: qdrant, path: {root}/qdrant, collection: kb, embedding: {embed} }}',
    "pgvector": '  - {{ name: "pgvector:kb-v1", kind: pgvector, dsn: "{dsn}", table: documents, embedding: {embed2} }}',
}


def write_config(
    root: Path,
    backends: Sequence[str],
    dsn: str | None = None,
    adapter: str = "",
    base_model: str = "Qwen/Qwen2.5-0.5B",
    unlearn: str = "exact",
    embed: str = EMBED_MODEL,
    embed2: str = EMBED_MODEL_2,
) -> Path:
    stores = "\n".join(
        STORE_LINES[b].format(root=root, dsn=dsn or "", embed=embed, embed2=embed2)
        for b in backends
    )
    backing = BACKEND_NAMES[backends[0]]
    cfg = root / "tombstone.yaml"
    cfg.write_text(
        CONFIG.format(
            root=root,
            stores=stores,
            backing=backing,
            adapter=adapter,
            base_model=base_model,
            unlearn=unlearn,
        )
    )
    return cfg


def reset_dir(p: Path) -> Path:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)
    return p


def save_results(kind: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": kind,
        "generated": now_tag(),
        "git": git_head(),
        "machine": machine(),
        **payload,
    }
    path = RESULTS / f"{kind}-{payload['generated']}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    latest = RESULTS / f"{kind}-latest.json"
    latest.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def cost_add(key: str, seconds: float, extra: dict[str, Any] | None = None) -> None:
    """Append compute spend to bench/cost.json."""
    path = ROOT / "bench" / "cost.json"
    data: dict[str, Any] = (
        json.loads(path.read_text())
        if path.is_file()
        else {"entries": [], "total_cpu_hours": 0.0, "gpu_minutes": 0.0}
    )
    data["entries"].append(
        {"key": key, "seconds": round(seconds, 1), "when": now_tag(), **(extra or {})}
    )
    data["total_cpu_hours"] = round(sum(e["seconds"] for e in data["entries"]) / 3600.0, 3)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def stable_sample(items: Sequence[Any], n: int, seed: int) -> list[Any]:
    import random

    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    return items[:n]


def chunk_text(text: str, size: int = 220) -> list[str]:
    """Deterministic chunking without LangChain: split on sentence boundaries into ~size chars."""
    import re

    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    cur = ""
    for s in sents:
        if cur and len(cur) + 1 + len(s) > size:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out or [text]


def content_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:24]
