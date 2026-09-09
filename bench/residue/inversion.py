"""Vec2Text inversion of vectors recovered from soft-deleted storage (optional, [inversion] extra).

Credit: *Ghost Vectors* (arXiv 2606.18497) for the attack; ``vec2text`` (Morris et al. 2023) for
the inverter. The published inverter is trained for GTR-base embeddings, so this experiment
builds a separate Chroma index with ``sentence-transformers/gtr-t5-base`` (768 dims), performs the
app's native delete(), recovers each subject's full vector from the HNSW segment file by locating
its 32-dim fingerprint and reading the 768 floats that follow, inverts it, and checks whether the
canary substring is recovered. When ``vec2text`` is not installed this writes a result file that
says inversion was not run — never a fabricated rate.

    uv run python bench/residue/inversion.py --subjects 20
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import WORK, chunk_text, embedder, reset_dir, save_results, write_config
from corpus.build import load_corpus
from tombstone.commands.init import run_init
from tombstone.commands.trace import run_trace
from tombstone.lineage.stamp import stamp
from tombstone.model.artifacts import ArtifactKind
from tombstone.registry import Runtime
from tombstone.stores._vector import VectorBackendBase
from tombstone.util import fingerprint_bytes

GTR = "gtr-t5-base"


def recover_vector(files: list[Path], fingerprint_hex: str, dims: int) -> list[float] | None:
    pat = fingerprint_bytes(fingerprint_hex)
    for f in files:
        data = f.read_bytes()
        i = data.find(pat)
        if i >= 0 and i + 4 * dims <= len(data):
            return list(struct.unpack(f"<{dims}f", data[i : i + 4 * dims]))
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=20)
    ns = ap.parse_args(argv)
    try:
        import vec2text
    except ImportError:
        save_results(
            "inversion",
            {
                "attempted": 0,
                "recovered": 0,
                "inverted": 0,
                "canary_recovered": 0,
                "canary_rate": None,
                "mean_overlap": None,
                "note": "vec2text not installed; inversion was not run",
            },
        )
        print("vec2text not installed; wrote a 'not run' result", file=sys.stderr)
        return 0
    import torch
    import vec2text

    docs = load_corpus()
    subjects = [f"S-{i:04d}" for i in range(1, ns.subjects + 1)]
    root = reset_dir(WORK / "inversion")
    run_init(root)
    cfg = write_config(root, ["chroma"], embed=GTR)
    rt = Runtime.shared(cfg)
    emb = embedder(GTR)
    store = rt.store("chroma:kb-v2", dims=emb.dims)
    assert isinstance(store, VectorBackendBase)
    capture = rt.capture()
    pepper = rt.pepper()
    canary_docs = [d for d in docs if d.subject in subjects and d.canary]
    filler = [d for d in docs if d.subject == "PUBLIC"][:400]
    for d in canary_docs + filler:
        md = stamp({"source": d.source}, d.subject, d.source, "default", pepper=pepper)
        capture.ensure_source(md, d.text)
        # short chunks: the published inverter was trained on 32-token inputs
        pieces = [d.canary.sentence] if d.canary else chunk_text(d.text, 120)[:1]
        keys = [f"{d.doc_id}#{i}" for i in range(len(pieces))]
        store.add(
            capture.prepare_embeds(
                store.name, emb.name, keys, emb.embed(pieces), [md] * len(pieces), pieces
            )
        )
    corrector = vec2text.load_pretrained_corrector("gtr-base")
    attempted = recovered = inverted = canary_hits = 0
    overlaps: list[float] = []
    t0 = time.time()
    for d in canary_docs:
        t, _ = run_trace(rt, d.subject, with_store_gaps=False)
        refs = [a for a in t.artifacts if a.kind is ArtifactKind.EMBED and a.embedding_fingerprint]
        store.native_delete([r.store_key for r in refs])
        for ref in refs:
            attempted += 1
            vec = recover_vector(store.persisted_files(), ref.embedding_fingerprint or "", emb.dims)
            if vec is None:
                continue
            recovered += 1
            text = vec2text.invert_embeddings(
                embeddings=torch.tensor([vec]), corrector=corrector, num_steps=20
            )[0]
            inverted += 1
            hit = d.canary.token.replace("-", "")[:8] in text.replace("-", "").upper()
            canary_hits += int(hit)
            a, b = set(text.lower().split()), set(d.canary.sentence.lower().split())
            overlaps.append(len(a & b) / max(1, len(b)))
    rt.close()
    payload = {
        "attempted": attempted,
        "recovered": recovered,
        "inverted": inverted,
        "canary_recovered": canary_hits,
        "canary_rate": canary_hits / max(1, inverted) if inverted else None,
        "mean_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
        "embedding": GTR,
        "wall_s": round(time.time() - t0, 1),
        "note": "inverter: vec2text gtr-base corrector, 20 steps; vectors recovered from chroma segment files after native delete()",
    }
    save_results("inversion", payload)
    print(payload, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
