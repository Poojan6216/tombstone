"""7.3 — paraphrased cache hits: after erasure, query the semantic cache with 20 paraphrases; a
cache entry whose parents did not include the subject's chunks (an answer generated from a
neighbour that quoted them) survives. Mitigation: purge the neighbourhood (k), with its cost."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import WORK
from adversarial._harness import Pipeline
from corpus.build import load_corpus
from tombstone.lineage.stamp import K_CHUNK
from tombstone.stores.cache_semantic import SemanticCache

PARAPHRASES = [
    "what is the membership number of {name}",
    "{name} membership number",
    "tell me {name}'s account code",
    "which reference is {name} enrolled under",
    "customer key for {name}",
    "policy number quoted for {name}",
    "{name} loyalty identifier",
    "what code did {name} confirm",
    "{name} account reference please",
    "number associated with {name}",
    "find the identifier of {name}",
    "{name} record id",
    "give me {name}'s membership",
    "the reference for customer {name}",
    "id for {name}",
    "{name} customer number",
    "membership id {name}",
    "what was {name} enrolled as",
    "{name} number",
    "{name}: identifier",
]


def run(n_subjects: int) -> dict[str, Any]:
    docs = load_corpus()
    subjects = [f"S-{i:04d}" for i in range(1, n_subjects + 1)]
    subject_docs = [d for d in docs if d.subject in subjects]
    out = {}
    for purge_k in (0, 3):
        p = Pipeline(WORK / "attacks" / f"s73-k{purge_k}")
        p.ingest(subject_docs, capture=True)
        cache = p.rt.store("semantic-cache")
        assert isinstance(cache, SemanticCache)
        cache.purge_k = purge_k
        # the app answered questions about each subject: the answer quotes the retrieved chunk
        # (which contains the canary) but the *parents* recorded are the top-3 chunks — for a
        # neighbour's question the subject's chunk can be quoted without being a parent
        canaries = {d.subject: d.canary for d in subject_docs if d.canary}
        entry_subject: dict[str, str] = {}  # cache key → the subject the question was about
        for s, can in canaries.items():
            for q in (f"what is the membership number of {can.name}", f"tell me about {can.name}"):
                hits = p.store.query(p.emb.embed([q])[0], 3)
                answer = " ".join((h.document or "") for h in hits)[:400]
                parents = [
                    str(h.metadata.get(K_CHUNK)) for h in hits[1:] if h.metadata.get(K_CHUNK)
                ]  # neighbour-only parents
                cache.update(q, answer, parents)
                from tombstone.util import sha256_hex

                entry_subject[sha256_hex(q)[:32]] = s
        total_entries = cache.count()
        leaked = 0
        collateral = 0
        erased: set[str] = set()
        for s, can in canaries.items():
            before = set(cache.backing.all_keys())
            p.erase(s)
            erased.add(s)
            after = set(cache.backing.all_keys())
            # entries removed that belonged to a subject not (yet) erased: the cost of purge_k
            collateral += sum(1 for k in before - after if entry_subject.get(k) not in erased)
            hit = False
            for tpl in PARAPHRASES:
                r = cache.lookup(tpl.format(name=can.name))
                if r is not None and can.token in r[0]:
                    hit = True
                    break
            leaked += int(hit)
        remaining = cache.count()
        out[f"k{purge_k}"] = {
            "subjects": len(canaries),
            "leaked": leaked,
            "rate": leaked / max(1, len(canaries)),
            "cache_entries_before": total_entries,
            "cache_entries_after": remaining,
            "collateral_purged": collateral,
        }
        p.close()
    return {
        "id": "7.3",
        "name": "paraphrased semantic-cache hits",
        "survives": "cached answers quoting the subject via a neighbour's question",
        "rate": out["k0"]["rate"],
        "rate_text": f"purge_k=0: {out['k0']['leaked']}/{out['k0']['subjects']} subjects leak through 20 paraphrases; purge_k=3: {out['k3']['leaked']}/{out['k3']['subjects']} (collateral: {out['k3']['collateral_purged']} unrelated entries purged of {out['k3']['cache_entries_before']})",
        "mitigation": "SemanticCache(purge_k=3): invalidate the k nearest cache neighbours of every erased entry — measured collateral cost above",
        "detail": out,
    }
