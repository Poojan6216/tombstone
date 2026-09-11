"""The serving composition: a SISA-style prediction-level ensemble of shard adapters.

Why not a weight merge: merging the shard adapters' LoRA deltas (linear sum, linear average,
or concatenation) destroyed memorised facts in our measurement — each shard alone extracted its
canaries, every merge extracted 0–1 of 6 (``bench/results/composition-*.json``). SISA
(Bourtoule et al. 2021) aggregates *predictions*, and that is what this does: every active shard
adapter scores the next token and the ensemble follows the most confident shard
(likelihood-weighted mixture). The shard that memorised a fact is near-certain at the fact's tokens,
so memorisation survives composition; excluding a shard removes its knowledge immediately, and
retraining it without a subject is exact unlearning at one shard's cost. Shards are weighted by
the likelihood they assign to the context (a posterior over experts), not by per-token
confidence — see the comment in ``_mixture`` for the measurement that ruled that out.

Cost: one forward pass per active shard per token, with a KV cache per shard so each step
costs one position rather than the whole sequence again (measured 2.1x on the benchmark's own
prompts; see ``generate_greedy``). The shard weights come from the context's cumulative
log-likelihood, which is a running sum, so it is accumulated across steps instead of recomputed.
Utility on unrelated text is the gated distribution's perplexity.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tombstone.train._torch import device, load_base, require_torch
from tombstone.util import atomic_write_text

MANIFEST = "serving.json"


def write_serving_manifest(
    adapters_dir: Path, serving_dir: Path, base_model: str, exclude: Sequence[int]
) -> dict[str, Any]:
    shard_dirs = sorted(
        p.name
        for p in adapters_dir.glob("shard-*")
        if p.is_dir()
        and (p / "adapter_config.json").is_file()
        and int(p.name.split("-")[1]) not in set(exclude)
    )
    if not shard_dirs:
        raise ValueError("no shard adapters to compose")
    if serving_dir.exists():
        shutil.rmtree(serving_dir)
    serving_dir.mkdir(parents=True)
    from tombstone.train.finetune import adapter_hash

    meta = {
        "composition": "SISA prediction ensemble, max-confidence gating",
        "base_model": base_model,
        "shards": shard_dirs,
        "excluded": sorted(set(exclude)),
        "shard_hashes": {d: adapter_hash(adapters_dir / d) for d in shard_dirs},
    }
    atomic_write_text(serving_dir / MANIFEST, json.dumps(meta, indent=1, sort_keys=True))
    atomic_write_text(
        serving_dir / "adapter_config.json",
        json.dumps({"tombstone_ensemble": True, "shards": shard_dirs}),
    )
    return meta


def is_ensemble_dir(path: Path) -> bool:
    return (path / MANIFEST).is_file()


class ShardEnsemble:
    """Greedy generation and token log-probs over gated shard adapters."""

    def __init__(self, base_model: str, adapters_dir: Path, shard_names: Sequence[str]) -> None:
        from peft import PeftModel

        self.torch = require_torch()
        self.tok, model = load_base(base_model)
        self.names = list(shard_names)
        self.pm = PeftModel.from_pretrained(
            model, str(adapters_dir / self.names[0]), adapter_name=self.names[0]
        )
        for n in self.names[1:]:
            self.pm.load_adapter(str(adapters_dir / n), adapter_name=n)
        self.pm.eval()

    @classmethod
    def load(
        cls, base_model: str, serving_dir: Path, adapters_dir: Path | None = None
    ) -> ShardEnsemble:
        meta = json.loads((serving_dir / MANIFEST).read_text())
        return cls(base_model, adapters_dir or serving_dir.parent, meta["shards"])

    # --- core: likelihood-weighted mixture of shard experts ------------------------------------
    #
    # Per-position max-confidence gating was measured first and rejected: an over-fitted shard
    # is "confident" everywhere, so it won the gate on unrelated text (held-out perplexity in
    # the hundreds of thousands) and on other shards' canaries (2/6 extracted). The mixture
    # below weights each shard by the likelihood it assigns to the context so far — the shard
    # that memorised a document recognises its own prefix — and on unrelated text the weights
    # are near-uniform, so utility is that of an ordinary mixture.

    def _shard_logprobs(self, input_ids: Any, attention_mask: Any) -> list[Any]:
        torch = self.torch
        out = []
        with torch.no_grad():
            for n in self.names:
                self.pm.set_adapter(n)
                logits = self.pm(input_ids=input_ids, attention_mask=attention_mask).logits.float()
                out.append(torch.log_softmax(logits, dim=-1))  # [batch, seq, vocab]
        return out

    def _mixture(self, lps: list[Any], input_ids: Any) -> Any:
        """[batch, seq, vocab] mixture log-probs with shard weights from the sequence likelihood."""
        torch = self.torch
        targets = input_ids[:, 1:]
        loglik = []
        for lp in lps:
            picked = lp[:, :-1].gather(2, targets.unsqueeze(2)).squeeze(2)  # [batch, seq-1]
            loglik.append(picked.sum(dim=1))  # [batch]
        w = torch.log_softmax(torch.stack(loglik, dim=0), dim=0)  # [shards, batch]
        stacked = torch.stack(lps, dim=0)  # [shards, batch, seq, vocab]
        return torch.logsumexp(stacked + w[:, :, None, None], dim=0)

    def _gated_logprobs(self, input_ids: Any, attention_mask: Any) -> Any:
        return self._mixture(self._shard_logprobs(input_ids, attention_mask), input_ids)

    def token_logprobs(self, text: str, max_len: int = 128) -> list[float]:
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device())
        lp = self._gated_logprobs(enc["input_ids"], enc["attention_mask"])[0, :-1]
        targets = enc["input_ids"][0, 1:]
        picked = lp.gather(1, targets.unsqueeze(1)).squeeze(1)
        return [float(x) for x in picked.tolist()]

    def generate_greedy(self, prefix: str, max_new_tokens: int = 16) -> str:
        """Greedy decode under the gated mixture, one position per shard per step.

        Equivalent to calling ``_gated_logprobs`` on the whole sequence at every step (see
        ``generate_greedy_uncached``, kept as the reference the equivalence test checks against),
        but that re-encodes the entire prefix for all N shards on every token. Measured on the
        benchmark's own canary prompt (9-token prefix, 16 new tokens, 3 shards, CPU): 11.7s
        cached against 24.1s uncached, so 2.1x. The position ratio alone would predict more; at
        these prefix lengths the per-step cost is dominated by the N adapter switches and model
        calls, which caching does not remove, so 2.1x is what it actually buys.

        Two things make the incremental form exact. The mixture only ever reads its last
        position here, so only the last row of each shard's log-probs is needed; and the shard
        weights come from the context's *cumulative* log-likelihood, which is a running sum — after
        a token is chosen, each shard's accumulator gains exactly the log-prob that shard assigned
        to it, which is the value already computed to choose it.
        """
        torch = self.torch
        enc = self.tok(prefix, return_tensors="pt").to(device())
        ids, mask = enc["input_ids"], enc["attention_mask"]
        eos = self.tok.eos_token_id

        caches: list[Any] = []
        loglik: list[Any] = []  # per shard, cumulative log-likelihood of the context
        last: list[Any] = []  # per shard, log-probs for the next token
        targets = ids[:, 1:]
        with torch.no_grad():
            for n in self.names:
                self.pm.set_adapter(n)
                o = self.pm(input_ids=ids, attention_mask=mask, use_cache=True)
                lp = torch.log_softmax(o.logits.float(), dim=-1)
                caches.append(o.past_key_values)
                loglik.append(lp[:, :-1].gather(2, targets.unsqueeze(2)).squeeze(2).sum(dim=1))
                last.append(lp[:, -1])

        out: list[int] = []
        for _ in range(max_new_tokens):
            w = torch.log_softmax(torch.stack(loglik, dim=0), dim=0)  # [shards, batch]
            mixed = torch.logsumexp(torch.stack(last, dim=0) + w[:, :, None], dim=0)
            nxt = int(torch.argmax(mixed[0]).item())
            if eos is not None and nxt == eos:
                break
            out.append(nxt)
            if len(out) == max_new_tokens:
                break
            step = torch.tensor([[nxt]], device=ids.device)
            mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)], 1)
            with torch.no_grad():
                for i, n in enumerate(self.names):
                    loglik[i] = loglik[i] + last[i][:, nxt]
                    self.pm.set_adapter(n)
                    o = self.pm(
                        input_ids=step,
                        attention_mask=mask,
                        past_key_values=caches[i],
                        use_cache=True,
                    )
                    caches[i] = o.past_key_values
                    last[i] = torch.log_softmax(o.logits.float(), dim=-1)[:, -1]
        return str(self.tok.decode(out, skip_special_tokens=True))

    def generate_greedy_uncached(self, prefix: str, max_new_tokens: int = 16) -> str:
        """The straightforward form: re-score the whole sequence every step.

        Kept only as the reference implementation that ``generate_greedy`` is tested against.
        """
        torch = self.torch
        enc = self.tok(prefix, return_tensors="pt").to(device())
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        eos = self.tok.eos_token_id
        out: list[int] = []
        for _ in range(max_new_tokens):
            lp = self._gated_logprobs(ids, mask)[0, -1]
            nxt = int(torch.argmax(lp).item())
            if eos is not None and nxt == eos:
                break
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
            mask = torch.cat(
                [mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)], dim=1
            )
        return str(self.tok.decode(out, skip_special_tokens=True))

    # --- duck-typing for extract.py / mia.py ----------------------------------------------------

    def generate(
        self, input_ids: Any = None, attention_mask: Any = None, max_new_tokens: int = 16, **_: Any
    ) -> Any:
        torch = self.torch
        prefix = self.tok.decode(input_ids[0], skip_special_tokens=True)
        text = self.generate_greedy(prefix, max_new_tokens)
        new = self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            input_ids.device
        )
        return torch.cat([input_ids, new], dim=1)

    def __call__(self, input_ids: Any = None, attention_mask: Any = None, **_: Any) -> Any:
        # already log-probs; a further log_softmax is idempotent
        return SimpleNamespace(logits=self._gated_logprobs(input_ids, attention_mask))

    def eval(self) -> ShardEnsemble:
        return self
