"""Generate docs/demo.md — the three demos from §2, every line captured from a real run.

Demo 1  native delete() then `tombstone verify --subject … --after-native-delete`
Demo 2  `tombstone erase` on self-hosted stores + caches + dataset + sharded adapter → VERIFIED
Demo 3  managed pgvector role (no file read / maintenance rights), unsharded adapter with
        approximate unlearning, and a subject mentioned in other subjects' documents → exit 2

Adapters come from bench/_work/unlearn (run bench/unlearn/run_unlearn.py first); without them
the model rows are reported as not run rather than faked.

    uv run python bench/demo.py [--subject S-0042] [--no-model]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ROOT,
    WORK,
    chunk_text,
    embedder,
    fresh_pg_database,
    pg_dsn,
    reset_dir,
    write_config,
)
from corpus.build import Doc, load_corpus
from tombstone.commands.init import run_init
from tombstone.lineage.stamp import K_CHUNK, stamp
from tombstone.model.artifacts import ArtifactKind
from tombstone.registry import Runtime
from tombstone.stores._vector import VectorBackendBase
from tombstone.train.dataset import DatasetStore, build_dataset

DOC = ROOT / "docs" / "demo.md"


def cli(args: list[str], cwd: Path) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, "-m", "tombstone", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "TOMBSTONE_LOG_LEVEL": "error"},
    )
    return p.returncode, (
        p.stdout + ("\n" + p.stderr if p.returncode not in (0, 2) else "")
    ).rstrip()


def build_app(
    root: Path,
    docs: list[Doc],
    backends: list[str],
    dsn: str | None,
    adapters_src: Path | None,
    shards: int,
    unlearn: str,
    subjects: list[str],
) -> tuple[Runtime, dict[str, Any]]:
    reset_dir(root)
    run_init(root)
    adapter_yaml = ""
    if adapters_src is not None and adapters_src.exists():
        dst = root / "adapters"
        shutil.copytree(adapters_src, dst)
        adapter_yaml = f'  - {{ name: "lora/support-v3", kind: adapter, path: {dst}, shards: {shards}, base_model: Qwen/Qwen2.5-0.5B, dataset: ft-dataset }}'
    cfg = write_config(root, backends, dsn, adapter=adapter_yaml, unlearn=unlearn)
    rt = Runtime.shared(cfg)
    pepper = rt.pepper()
    capture = rt.capture()
    docstore = rt.store("docs")
    emb = {
        b: embedder("bge-small-en-v1.5" if b == "pgvector" else "all-MiniLM-L6-v2")
        for b in backends
    }
    stores = {
        b: rt.store(
            {
                "chroma": "chroma:kb-v2",
                "pgvector": "pgvector:kb-v1",
                "faiss": "faiss:kb-v1",
                "qdrant": "qdrant:kb",
            }[b],
            dims=emb[b].dims,
        )
        for b in backends
    }
    chunks: list[tuple[Any, str]] = []
    for d in docs:
        md = stamp(
            {"source": d.source},
            d.subject,
            d.source,
            "default",
            pepper=pepper,
            mentions=list(d.mentions),
        )
        src = capture.ensure_source(md, d.text)
        docstore.put(src.artifact_id, "source", d.text, md)  # type: ignore[attr-defined]
        pieces = chunk_text(d.text)
        for b, store in stores.items():
            assert isinstance(store, VectorBackendBase)
            keys = [f"{d.doc_id}#{i}" for i in range(len(pieces))]
            recs = capture.prepare_embeds(
                store.name,
                emb[b].name,
                keys,
                emb[b].embed(pieces),
                [md] * len(pieces),
                pieces,
                emb[b].embed,
            )
            store.add(recs)
            if b == backends[0]:
                for r in recs:
                    docstore.put(r.chunk_node.artifact_id, "chunk", r.document or "", r.metadata)  # type: ignore[attr-defined]
                    if d.subject in subjects:
                        chunks.append((r.chunk_node, r.document or ""))
    # a RAG call per demo subject, cached both ways
    from langchain_core.outputs import Generation

    exact = rt.store("exact-cache")
    semantic = rt.store("semantic-cache")
    primary = stores[backends[0]]
    for s in subjects:
        can = next(d.canary for d in docs if d.subject == s and d.canary)
        q = f"What was the membership number for {can.name}?"
        hits = primary.query(emb[backends[0]].embed([can.sentence])[0], 3)
        answer = " ".join((h.document or "") for h in hits)[:300]
        prompt = f"Context:\n{chr(10).join((h.document or '') for h in hits)}\n<!-- tombstone:chunks={','.join(str(h.metadata.get(K_CHUNK)) for h in hits)} -->\nQ: {q}"
        exact.update(prompt, "extractive-llm", [Generation(text=answer)])  # type: ignore[attr-defined]
        semantic.update(q, answer, [str(h.metadata.get(K_CHUNK)) for h in hits])  # type: ignore[attr-defined]
    # dataset rows for the demo subjects' chunks (the adapters were trained on the bench dataset;
    # the demo's dataset store mirrors the rows so TRAIN artifacts trace)
    build_dataset(capture, "ft-dataset", root / "train" / "manifest.json", chunks, shards=shards)
    if adapter_yaml:
        from tombstone.stores.adapter import AdapterStore

        a = rt.store("lora/support-v3")
        assert isinstance(a, AdapterStore)
        a.register_lineage(capture, DatasetStore("ft-dataset", root / "train" / "manifest.json"))
    return rt, {"stores": stores, "emb": emb, "cfg": cfg}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S-0042")
    ap.add_argument("--no-model", action="store_true")
    ns = ap.parse_args(argv)
    docs = load_corpus()
    subject = ns.subject
    # a compact corpus: the subject, 40 other subjects, 200 public docs
    others = [f"S-{i:04d}" for i in range(1, 41) if f"S-{i:04d}" != subject]
    keep = {subject, *others, "PUBLIC"}
    corpus = [d for d in docs if d.subject in keep][:1500]
    adapters = None if ns.no_model else WORK / "unlearn" / "adapters"
    dsn_base = pg_dsn()
    out: list[str] = [
        "# Demos",
        "",
        f"Generated by `uv run python bench/demo.py --subject {subject}` on {time.strftime('%Y-%m-%d')}; every block below is captured from that run. Subject ids are shown as HMACs; the raw id never reaches disk.",
        "",
    ]
    # ---- Demo 1 --------------------------------------------------------------------------------
    root = WORK / "demo1"
    dsn = fresh_pg_database(dsn_base, "tomb_demo1") if dsn_base else None
    backends = ["chroma", "pgvector"] if dsn else ["chroma", "faiss"]
    rt, h = build_app(root, corpus, backends, dsn, adapters, 16, "exact", [subject])
    from tombstone.commands.trace import run_trace

    t, _ = run_trace(rt, subject, with_store_gaps=False)
    embeds = [a for a in t.artifacts if a.kind is ArtifactKind.EMBED]
    for b, store in h["stores"].items():
        store.native_delete([a.store_key for a in embeds if a.store == store.name])
        rt.capture().record_native_delete(
            store.name, [a.store_key for a in embeds if a.store == store.name]
        )
    rt.store("docs").native_delete(
        [a.store_key for a in t.artifacts if a.kind in {ArtifactKind.SOURCE, ArtifactKind.CHUNK}]
    )  # type: ignore[attr-defined]
    rt.close()
    code, text = cli(
        [
            "verify",
            "--config",
            str(h["cfg"]),
            "--subject",
            subject,
            "--after-native-delete",
            "--no-store-scan",
        ],
        root,
    )
    out += [
        "## Demo 1 — `delete()` is a lie",
        "",
        "The app deleted the way everyone deletes: `vectorstore.delete(ids=[...])` on both indexes and the source rows dropped. Then:",
        "",
        "```text",
        f"$ tombstone verify --subject {subject} --after-native-delete",
        text,
        f"exit {code}",
        "```",
        "",
    ]
    # ---- Demo 2 --------------------------------------------------------------------------------
    root = WORK / "demo2"
    dsn = fresh_pg_database(dsn_base, "tomb_demo2") if dsn_base else None
    rt, h = build_app(root, corpus, backends, dsn, adapters, 16, "exact", [subject])
    t, _ = run_trace(rt, subject, with_store_gaps=False)
    rt.close()
    code, text = cli(
        [
            "erase",
            "--config",
            str(h["cfg"]),
            "--trace",
            t.trace_id,
            "--reason",
            "dsr-2026-0912",
            "--confirm",
            "--semantic",
        ],
        root,
    )
    out += [
        "## Demo 2 — the cascade, and what it can honestly claim",
        "",
        "```text",
        f"$ tombstone erase --trace {t.trace_id} --reason dsr-2026-0912 --confirm --semantic",
        text,
        f"exit {code}",
        "```",
        "",
        "`OUT_OF_SCOPE` is a count, and it is never zero, because there are always layers this tool cannot see. The receipt says so.",
        "",
    ]
    # ---- Demo 3 --------------------------------------------------------------------------------
    root = WORK / "demo3"
    dsn3 = None
    if dsn_base:
        dsn3 = fresh_pg_database(dsn_base, "tomb_demo3")
        import psycopg

        with psycopg.connect(dsn3, autocommit=True) as conn:
            conn.execute("DROP ROLE IF EXISTS managed_app")
            conn.execute("CREATE ROLE managed_app LOGIN PASSWORD 'managed'")
            conn.execute("GRANT USAGE, CREATE ON SCHEMA public TO managed_app")
    rt, h = (
        build_app(root, corpus, backends, dsn3, adapters, 1, "npo", [subject])
        if not dsn3
        else _demo3_managed(root, corpus, dsn3, adapters, subject)
    )
    t, _ = run_trace(rt, subject, with_store_gaps=False)
    rt.close()
    code, text = cli(
        [
            "erase",
            "--config",
            str(h["cfg"]),
            "--trace",
            t.trace_id,
            "--reason",
            "dsr-2026-0913",
            "--confirm",
        ],
        root,
    )
    out += [
        "## Demo 3 — the honest one",
        "",
        "Same subject, but the pgvector instance is reached through a role without file-read or maintenance rights (a managed database), the adapter was trained without sharding, and the subject is mentioned inside two other subjects' documents.",
        "",
        "```text",
        f"$ tombstone erase --trace {t.trace_id} --reason dsr-2026-0913 --confirm",
        text,
        f"exit {code}",
        "```",
        "",
        'A tool that prints "erasure complete" here would be lying to a regulator on the operator\'s behalf. Tombstone refuses.',
        "",
    ]
    DOC.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {DOC}", file=sys.stderr)
    return 0


def _demo3_managed(
    root: Path, corpus: list[Doc], dsn_owner: str, adapters: Path | None, subject: str
) -> tuple[Runtime, dict[str, Any]]:
    """Ingest as the owner, then switch the config to the managed role for the erase."""
    # two other subjects' documents mention the target (no NER: the app recorded it at ingest)
    from corpus.build import SEED as CSEED
    from corpus.build import canary_for

    can = canary_for(subject, CSEED)
    extra = [
        Doc(
            "S-0003-m",
            "S-0003",
            "support/S-0003/m.txt",
            f"Case note. The agent linked this case to the household of {can.name}. Follow-up scheduled.",
            (subject,),
            None,
            0,
        ),
        Doc(
            "S-0007-m",
            "S-0007",
            "support/S-0007/m.txt",
            f"Escalation record. {can.name} was named as the account holder on the shared address. No further action.",
            (subject,),
            None,
            0,
        ),
    ]
    rt, h = build_app(
        root, corpus + extra, ["chroma", "pgvector"], dsn_owner, adapters, 1, "npo", [subject]
    )
    import psycopg

    with psycopg.connect(dsn_owner, autocommit=True) as conn:
        conn.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON documents TO managed_app")
    base, _, dbname = dsn_owner.rpartition("/")
    host = base.split("@")[-1]
    managed = f"postgresql://managed_app:managed@{host}/{dbname}"
    cfg = h["cfg"]
    cfg.write_text(cfg.read_text().replace(f'dsn: "{dsn_owner}"', f'dsn: "{managed}"'))
    rt.close()
    rt = Runtime.shared(cfg)
    return rt, h


if __name__ == "__main__":
    raise SystemExit(main())
