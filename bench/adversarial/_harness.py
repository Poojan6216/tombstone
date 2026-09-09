"""Shared harness for the attack strategies: a small pipeline over the bench corpus subset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (
    BACKEND_NAMES,
    EMBED_MODEL,
    chunk_text,
    embedder,
    reset_dir,
    write_config,
)
from corpus.build import Doc
from tombstone.commands.erase import run_erase
from tombstone.commands.init import run_init
from tombstone.commands.trace import run_trace
from tombstone.lineage.stamp import stamp
from tombstone.registry import Runtime
from tombstone.stores._vector import VectorBackendBase


class Pipeline:
    """One backend (default FAISS: fast, physical) with docstore + caches, built from docs."""

    def __init__(
        self, root: Path, backend: str = "faiss", extra_yaml: str = "", pg_dsn: str | None = None
    ) -> None:
        reset_dir(root)
        run_init(root)
        self.root = root
        self.backend = backend
        self.cfg = write_config(root, [backend], pg_dsn, adapter=extra_yaml)
        self.rt = Runtime.shared(self.cfg)
        self.pepper = self.rt.pepper()
        self.emb = embedder(EMBED_MODEL)
        self.store = self.rt.store(BACKEND_NAMES[backend], dims=self.emb.dims)
        assert isinstance(self.store, VectorBackendBase)
        self.capture = self.rt.capture()
        self.docstore = self.rt.store("docs")
        self.keys_by_doc: dict[str, list[str]] = {}

    def ingest(
        self,
        docs: list[Doc],
        capture: bool = True,
        subject_override: dict[str, str] | None = None,
        derived_from: dict[str, str] | None = None,
    ) -> None:
        """Ingest with lineage (capture=True) or straight into the store (capture=False)."""
        keys, texts, mds = [], [], []
        for d in docs:
            subj = (subject_override or {}).get(d.doc_id, d.subject)
            md = stamp(
                {"source": d.source},
                subj,
                d.source,
                "default",
                pepper=self.pepper,
                mentions=list(d.mentions),
                derived_from=(derived_from or {}).get(d.doc_id),
            )
            if capture:
                src = self.capture.ensure_source(md, d.text)
                self.docstore.put(src.artifact_id, "source", d.text, md)  # type: ignore[attr-defined]
            for i, ch in enumerate(chunk_text(d.text)):
                key = f"{d.doc_id}#{i}"
                keys.append(key)
                texts.append(ch)
                mds.append(md)
                self.keys_by_doc.setdefault(d.doc_id, []).append(key)
        for i in range(0, len(keys), 512):
            sl = slice(i, i + 512)
            vecs = self.emb.embed(texts[sl])
            if capture:
                recs = self.capture.prepare_embeds(
                    self.store.name,
                    self.emb.name,
                    keys[sl],
                    vecs,
                    mds[sl],
                    texts[sl],
                    self.emb.embed,
                )
                self.store.add(recs)
            else:
                from tombstone.lineage.capture import EmbedRecord

                recs = [
                    EmbedRecord(k, v, {**m, "legacy": True}, t, None, None)
                    for k, v, m, t in zip(keys[sl], vecs, mds[sl], texts[sl], strict=True)
                ]  # type: ignore[arg-type]
                self.store._add(recs)

    def erase(self, subject: str, accept_gaps: bool = False) -> tuple[int, dict[str, Any], Any]:
        t, _ = run_trace(self.rt, subject)
        code, _text, data = run_erase(
            self.rt, t.trace_id, f"attack-{subject}", confirm=True, accept_gaps=accept_gaps
        )
        return code, data, t

    def canary_hits(self, canary_token: str, k: int = 10) -> int:
        """How many retrievable chunks still contain the canary token (by top-k over probes)."""
        hits = set()
        for q in (
            canary_token,
            f"membership number {canary_token}",
            canary_token.replace("-", " "),
        ):
            for h in self.store.query(self.emb.embed([q])[0], k):
                if h.document and canary_token in h.document:
                    hits.add(h.key)
        return len(hits)

    def close(self) -> None:
        self.rt.close()
