"""Membership inference: loss-based and Min-K% over the subject's examples vs a held-out
reference set of matched length. AUC with a bootstrap confidence interval (fixed-sample
bootstrap over the score pairs — fine here: nobody re-peeks at an MIA).

References: Yeom et al. 2018 (loss attack); Shi et al. 2024, *Detecting Pretraining Data from
Large Language Models* (Min-K% Prob). Verification of unlearning is fragile (arXiv 2408.00929):
an AUC at chance is evidence of absence only against these two attacks.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tombstone.train._torch import device, require_torch


@dataclass(frozen=True, slots=True)
class MIAResult:
    attack: str
    auc: float
    ci_low: float
    ci_high: float
    n_members: int
    n_nonmembers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack": self.attack,
            "auc": self.auc,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_members": self.n_members,
            "n_nonmembers": self.n_nonmembers,
        }

    @property
    def at_chance(self) -> bool:
        return self.ci_low <= 0.5 <= self.ci_high


def token_logprobs(tok: Any, model: Any, text: str, max_len: int = 128) -> list[float]:
    torch = require_torch()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device())
    with torch.no_grad():
        logits = model(**enc).logits[0, :-1]
        targets = enc["input_ids"][0, 1:]
        lp = torch.log_softmax(logits.float(), dim=-1)
        picked = lp.gather(1, targets.unsqueeze(1)).squeeze(1)
    return [float(x) for x in picked.tolist()]


def loss_score(lps: Sequence[float]) -> float:
    """Higher = more member-like. Negative mean NLL."""
    return sum(lps) / max(1, len(lps))


def mink_score(lps: Sequence[float], k: float = 0.2) -> float:
    """Min-K%: mean log-prob of the k fraction of lowest-probability tokens. Higher = member."""
    if not lps:
        return 0.0
    n = max(1, math.ceil(len(lps) * k))
    lowest = sorted(lps)[:n]
    return sum(lowest) / n


def auc(members: Sequence[float], nonmembers: Sequence[float]) -> float:
    """Mann-Whitney AUC: P(score_member > score_nonmember)."""
    if not members or not nonmembers:
        return 0.5
    wins = 0.0
    for m in members:
        for n in nonmembers:
            if m > n:
                wins += 1.0
            elif m == n:
                wins += 0.5
    return wins / (len(members) * len(nonmembers))


def bootstrap_auc(
    members: Sequence[float], nonmembers: Sequence[float], n_boot: int = 500, seed: int = 0
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    point = auc(members, nonmembers)
    samples = []
    for _ in range(n_boot):
        ms = [members[rng.randrange(len(members))] for _ in members]
        ns = [nonmembers[rng.randrange(len(nonmembers))] for _ in nonmembers]
        samples.append(auc(ms, ns))
    samples.sort()
    lo = samples[int(0.025 * (n_boot - 1))]
    hi = samples[int(0.975 * (n_boot - 1))]
    return point, lo, hi


def membership_inference(
    tok: Any, model: Any, member_texts: Sequence[str], reference_texts: Sequence[str], seed: int = 0
) -> dict[str, MIAResult]:
    """Both attacks; the reference set is truncated/matched to the members' length."""
    n = min(len(member_texts), len(reference_texts))
    members = list(member_texts)[:n]
    refs = list(reference_texts)[:n]
    m_lps = [token_logprobs(tok, model, t) for t in members]
    r_lps = [token_logprobs(tok, model, t) for t in refs]
    out: dict[str, MIAResult] = {}
    for name, fn in (("loss", loss_score), ("mink", mink_score)):
        ms = [fn(x) for x in m_lps]
        ns = [fn(x) for x in r_lps]
        point, lo, hi = bootstrap_auc(ms, ns, seed=seed)
        out[name] = MIAResult(name, point, lo, hi, len(ms), len(ns))
    return out


def perplexity(tok: Any, model: Any, texts: Sequence[str]) -> float:
    total = 0.0
    count = 0
    for t in texts:
        lps = token_logprobs(tok, model, t)
        total += -sum(lps)
        count += len(lps)
    return math.exp(total / max(1, count))
