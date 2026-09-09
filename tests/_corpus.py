"""Load the committed small corpus, filling canaries from the seed at load time."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tombstone.train.canaries import Canary, canary_for

FIXTURE = Path(__file__).parent / "fixtures" / "corpus_small" / "docs.json"


@dataclass(frozen=True, slots=True)
class CorpusDoc:
    doc_id: str
    subject: str
    source: str
    text: str
    mentions: tuple[str, ...]
    canary: Canary | None


def load_corpus() -> list[CorpusDoc]:
    blob = json.loads(FIXTURE.read_text(encoding="utf-8"))
    seed = int(blob["seed"])
    out: list[CorpusDoc] = []
    for d in blob["docs"]:
        canary = canary_for(d["subject"], seed) if d["canary"] else None
        text = d["text"]
        text = text.replace("{canary}", canary.sentence if canary else "")
        if d["mentions"]:
            text = text.replace("{mentioned}", canary_for(d["mentions"][0], seed).name)
        out.append(
            CorpusDoc(
                d["doc_id"],
                d["subject"],
                d["source"],
                " ".join(text.split()),
                tuple(d["mentions"]),
                canary,
            )
        )
    return out


def subjects() -> list[str]:
    return sorted({d.subject for d in load_corpus() if d.subject != "PUBLIC"})
