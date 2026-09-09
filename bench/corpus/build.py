"""6.1 — the benchmark corpus.

Public part: **AG News** (Zhang, Zhao & LeCun 2015; HF dataset ``ag_news``, train split), 2,000
passages chosen by a seeded shuffle. Chosen because it is redistributable by reference, small,
plain-text, and topically clustered (four classes) so same-cluster controls for the drift
protocol are meaningful. The corpus is not committed: ``manifest.json`` holds ids and SHA-256
hashes; ``load_corpus()`` re-downloads AG News and verifies every hash.

Synthetic part: 200 subjects ``S-0001``…``S-0200``, each with 3–8 documents. One document per
subject carries the subject's canary sentence (``tombstone.train.canaries``, from the seed; the
manifest stores only the canary hash). ~4% of subject documents mention another subject
(``mentions``), for the third-party experiments. Deterministic from ``SEED``.

    uv run python bench/corpus/build.py            # writes bench/corpus/manifest.json
    uv run python bench/corpus/build.py --check    # rebuild and compare byte-for-byte
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tombstone.train.canaries import Canary, canary_for
from tombstone.util import sha256_hex

SEED = 20260908
N_PUBLIC = 2000
N_SUBJECTS = 200
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
CACHE = HERE / "_cache"

_TOPICS = [
    "a refund request",
    "a shipping delay",
    "an address change",
    "a password reset",
    "a subscription pause",
    "a warranty claim",
    "a data export request",
    "an invoice dispute",
    "a duplicate charge",
    "a cancelled order",
    "an account merge",
    "a name correction",
]
_OPENERS = [
    "Support transcript.",
    "Case summary.",
    "Follow-up note.",
    "Escalation record.",
    "Chat log excerpt.",
    "Email thread summary.",
]
_MIDDLES = [
    "The agent reviewed the account history and proposed a resolution within two business days.",
    "After verifying the identity of the caller, the agent updated the case and set a reminder.",
    "The customer asked for written confirmation and the agent attached the policy excerpt.",
    "A supervisor approved the exception and the ticket moved to the billing queue.",
    "The agent explained the timeline and the customer agreed to wait for the next cycle.",
    "Two earlier tickets were linked and the duplicate was closed as resolved.",
]
_CLOSERS = [
    "The customer thanked the agent and the case was closed.",
    "The case remains open pending a callback.",
    "A satisfaction survey was sent after the call.",
    "No further action was required.",
]


@dataclass(frozen=True, slots=True)
class Doc:
    doc_id: str
    subject: str  # "PUBLIC" for AG News passages
    source: str
    text: str
    mentions: tuple[str, ...]
    canary: Canary | None
    cluster: int  # AG News label for public docs, subject index % 4 for synthetic


def _public_passages() -> list[tuple[str, int]]:
    """(text, label) for the AG News train split in dataset order (cached locally)."""
    cache = CACHE / "ag_news_train.jsonl"
    if cache.is_file():
        rows = [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines() if x.strip()]
        return [(r["text"], int(r["label"])) for r in rows]
    from datasets import load_dataset

    ds = load_dataset("ag_news", split="train")
    CACHE.mkdir(parents=True, exist_ok=True)
    with cache.open("w", encoding="utf-8") as fh:
        for r in ds:
            fh.write(json.dumps({"text": r["text"], "label": int(r["label"])}) + "\n")
    return [(str(r["text"]), int(r["label"])) for r in ds]


def _pick_public(rng: random.Random) -> list[Doc]:
    rows = _public_passages()
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    out: list[Doc] = []
    for i in idx[:N_PUBLIC]:
        text, label = rows[i]
        text = " ".join(text.split())
        out.append(Doc(f"AG-{i:06d}", "PUBLIC", f"agnews/train/{i}", text, (), None, label))
    return out


def _synthetic(rng: random.Random) -> list[Doc]:
    subjects = [f"S-{i:04d}" for i in range(1, N_SUBJECTS + 1)]
    canaries = {s: canary_for(s, SEED) for s in subjects}
    out: list[Doc] = []
    for si, sid in enumerate(subjects):
        n_docs = rng.randint(3, 8)
        canary_doc = rng.randrange(n_docs)
        for d in range(n_docs):
            can = canaries[sid]
            topic = rng.choice(_TOPICS)
            parts = [
                f"{rng.choice(_OPENERS)} The customer, {can.name}, wrote in about {topic}.",
                rng.choice(_MIDDLES),
            ]
            mentions: list[str] = []
            if d == canary_doc:
                parts.append(can.sentence)
            elif rng.random() < 0.08:
                other = rng.choice(subjects)
                if other != sid:
                    mentions.append(other)
                    parts.append(
                        f"The agent noted that this case is linked to the household of {canaries[other].name}."
                    )
            parts.append(rng.choice(_CLOSERS))
            text = " ".join(parts)
            out.append(
                Doc(
                    f"{sid}-{d}",
                    sid,
                    f"support/{sid}/{d}.txt",
                    text,
                    tuple(mentions),
                    can if d == canary_doc else None,
                    si % 4,
                )
            )
    return out


def build() -> list[Doc]:
    rng = random.Random(SEED)
    docs = _pick_public(rng) + _synthetic(rng)
    return docs


def manifest_for(docs: list[Doc]) -> dict[str, object]:
    entries = []
    for d in docs:
        entries.append(
            {
                "doc_id": d.doc_id,
                "subject": d.subject,
                "source": d.source,
                "sha256": sha256_hex(d.text),
                "chars": len(d.text),
                "cluster": d.cluster,
                "mentions": list(d.mentions),
                "canary_sha256": d.canary.token_hash if d.canary else None,
            }
        )
    body = {
        "version": 1,
        "seed": SEED,
        "public": {
            "dataset": "ag_news",
            "split": "train",
            "n": N_PUBLIC,
            "citation": "Zhang, Zhao, LeCun (2015)",
        },
        "subjects": N_SUBJECTS,
        "docs": entries,
    }
    body["manifest_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def load_corpus(verify: bool = True) -> list[Doc]:
    """Rebuild from seed + AG News and (optionally) verify against the committed manifest."""
    docs = build()
    if verify and MANIFEST.is_file():
        committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
        want = {e["doc_id"]: e["sha256"] for e in committed["docs"]}
        got = {d.doc_id: sha256_hex(d.text) for d in docs}
        if want != got:
            diff = [k for k in want if want.get(k) != got.get(k)]
            raise RuntimeError(
                f"corpus does not match manifest: {len(diff)} documents differ (first {diff[:3]})"
            )
    return docs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check", action="store_true", help="rebuild and compare with the committed manifest"
    )
    ns = ap.parse_args(argv)
    docs = build()
    m = manifest_for(docs)
    text = json.dumps(m, indent=1, sort_keys=True) + "\n"
    if ns.check:
        if not MANIFEST.is_file():
            print("no manifest committed", file=sys.stderr)
            return 1
        ok = MANIFEST.read_text(encoding="utf-8") == text
        print("manifest byte-identical" if ok else "manifest DIFFERS", file=sys.stderr)
        return 0 if ok else 1
    MANIFEST.write_text(text, encoding="utf-8")
    n_pub = sum(1 for d in docs if d.subject == "PUBLIC")
    n_subj = len({d.subject for d in docs if d.subject != "PUBLIC"})
    n_can = sum(1 for d in docs if d.canary)
    n_ment = sum(1 for d in docs if d.mentions)
    print(
        f"wrote {MANIFEST}: {len(docs)} docs ({n_pub} public, {n_subj} subjects, {n_can} canaries, {n_ment} mentions)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
