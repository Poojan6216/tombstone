"""Model operations behind an injectable interface, so the adapter store can be tested with a
fake and run for real with torch. The real implementation lives here and in the sibling modules."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from tombstone.train.canaries import Canary


class ModelOps(Protocol):
    def train_shard(
        self, dataset: Any, adapters_dir: Path, shard: int, cfg: Any
    ) -> dict[str, Any]: ...
    def compose(
        self, adapters_dir: Path, serving_dir: Path, base_model: str, exclude: Sequence[int]
    ) -> dict[str, Any]: ...
    def exact_unlearn(
        self, dataset: Any, adapters_dir: Path, serving_dir: Path, shard: int, cfg: Any
    ) -> dict[str, Any]: ...
    def approximate_unlearn(
        self,
        base_model: str,
        adapter_dir: Path,
        forget: Sequence[str],
        retain: Sequence[str],
        cfg: Any,
    ) -> dict[str, Any]: ...
    def extraction(
        self, base_model: str, adapter_dir: Path, prompts: Sequence[tuple[str, str]]
    ) -> tuple[int, int]: ...
    def mia(
        self, base_model: str, adapter_dir: Path, members: Sequence[str], reference: Sequence[str]
    ) -> dict[str, Any]: ...
    def perplexity(self, base_model: str, adapter_dir: Path, texts: Sequence[str]) -> float: ...


class TorchOps:
    """The real thing. Every method loads the base model (cached by transformers)."""

    def __init__(self, log: Any = None) -> None:
        self.log = log

    def train_shard(self, dataset: Any, adapters_dir: Path, shard: int, cfg: Any) -> dict[str, Any]:
        from tombstone.train.finetune import train_shards

        return train_shards(dataset, adapters_dir, cfg, shards=[shard], log=self.log)

    def compose(
        self, adapters_dir: Path, serving_dir: Path, base_model: str, exclude: Sequence[int]
    ) -> dict[str, Any]:
        from tombstone.train.finetune import compose_serving

        return compose_serving(adapters_dir, serving_dir, base_model, exclude)

    def exact_unlearn(
        self, dataset: Any, adapters_dir: Path, serving_dir: Path, shard: int, cfg: Any
    ) -> dict[str, Any]:
        from tombstone.train.unlearn import exact_unlearn

        return exact_unlearn(dataset, adapters_dir, serving_dir, shard, cfg, log=self.log)

    def approximate_unlearn(
        self,
        base_model: str,
        adapter_dir: Path,
        forget: Sequence[str],
        retain: Sequence[str],
        cfg: Any,
    ) -> dict[str, Any]:
        from tombstone.train.unlearn import approximate_unlearn

        return approximate_unlearn(
            base_model, adapter_dir, adapter_dir, forget, retain, cfg, log=self.log
        )

    def extraction(
        self, base_model: str, adapter_dir: Path, prompts: Sequence[tuple[str, str]]
    ) -> tuple[int, int]:
        from tombstone.train.extract import extract
        from tombstone.train.finetune import load_adapter_model

        tok, pm = load_adapter_model(base_model, adapter_dir)
        hits = 0
        for prefix, expected in prompts:
            out = extract(tok, pm, prefix)
            if _recovered(out, expected):
                hits += 1
        return hits, len(prompts)

    def mia(
        self, base_model: str, adapter_dir: Path, members: Sequence[str], reference: Sequence[str]
    ) -> dict[str, Any]:
        from tombstone.train.finetune import load_adapter_model
        from tombstone.train.mia import membership_inference

        tok, pm = load_adapter_model(base_model, adapter_dir)
        res = membership_inference(tok, pm, members, reference)
        return {k: v.to_dict() for k, v in res.items()}

    def perplexity(self, base_model: str, adapter_dir: Path, texts: Sequence[str]) -> float:
        from tombstone.train.finetune import load_adapter_model
        from tombstone.train.mia import perplexity

        tok, pm = load_adapter_model(base_model, adapter_dir)
        return perplexity(tok, pm, texts)


def _recovered(generated: str, expected: str, min_chars: int = 12) -> bool:
    flat = "".join(ch for ch in generated if ch.isalnum()).upper()
    exp = "".join(ch for ch in expected if ch.isalnum()).upper()
    return len(exp) >= 1 and exp[:min_chars] in flat


def prompts_from_canaries(canaries: Sequence[Canary]) -> list[tuple[str, str]]:
    return [(c.prefix, c.token) for c in canaries]


def prompts_from_texts(texts: Sequence[str], split: float = 0.7) -> list[tuple[str, str]]:
    """Generic memorisation prompts: the first ``split`` of each text, expecting the rest."""
    out: list[tuple[str, str]] = []
    for t in texts:
        cut = max(1, int(len(t) * split))
        prefix, cont = t[:cut].rstrip(), t[cut:].strip()
        if prefix and cont:
            out.append((prefix, cont))
    return out


def read_texts_jsonl(path: Path) -> list[str]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(str(d.get("text", "")))
    return out
