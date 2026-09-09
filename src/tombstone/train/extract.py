"""Canary extraction: greedy decode from the canary's prefix and check for the token."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tombstone.train._torch import device, require_torch
from tombstone.train.canaries import Canary, token_recovered


def extract(tok: Any, model: Any, prefix: str, max_new_tokens: int = 16) -> str:
    torch = require_torch()
    enc = tok(prefix, return_tensors="pt").to(device())
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    return str(tok.decode(gen[0][enc["input_ids"].shape[1] :], skip_special_tokens=True))


def canary_extraction_rate(
    tok: Any, model: Any, canaries: Sequence[Canary]
) -> tuple[int, int, list[bool]]:
    """(extracted, total, per-canary) with greedy decoding from each canary's prefix."""
    hits: list[bool] = []
    for c in canaries:
        out = extract(tok, model, c.prefix)
        hits.append(token_recovered(out, c))
    return sum(hits), len(canaries), hits
