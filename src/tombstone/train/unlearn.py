"""Unlearning: exact (shard retrain) and approximate (NPO, gradient difference).

exact
    SISA-style (Bourtoule et al. 2021, *Machine Unlearning*): the subject's examples live in one
    shard; drop them, retrain that shard's adapter from the base model, recompose. Cost is one
    shard, not the dataset. The data is not there, so nothing can be relearned from the weights.

npo
    Negative Preference Optimization (Zhang et al. 2024, arXiv 2404.05868): treat forget-set
    examples as dispreferred responses against the reference (pre-unlearning) model,
    L = -(2/β) E_forget[ log σ(-β (log π(x) - log π_ref(x))) ] + λ · NLL_retain.
    Bounded, unlike plain gradient ascent, so it does not collapse the model.

gradient_difference
    Gradient ascent on the forget set plus gradient descent on a retain set
    (Liu et al. 2022, *Continual learning and private unlearning*; also the GD baseline in TOFU,
    Maini et al. 2024, arXiv 2401.06121): L = -NLL_forget + λ · NLL_retain.

Hyperparameters are committed in ``bench/results/unlearn-grid-*.json`` (task 4.4).
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tombstone.train._torch import device, require_torch
from tombstone.train.dataset import DatasetStore
from tombstone.train.finetune import (
    TrainConfig,
    adapter_hash,
    compose_serving,
    load_adapter_model,
    train_lora,
)


@dataclass(frozen=True, slots=True)
class UnlearnConfig:
    method: str = "npo"  # npo | gradient_difference
    steps: int = 40
    lr: float = 1e-4
    beta: float = 0.1  # NPO temperature
    retain_weight: float = 1.0
    retain_size: int = 64
    batch_size: int = 4
    max_len: int = 96
    seed: int = 20260908

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- exact ------------------------------------------------------------------------------------


def exact_unlearn(
    dataset: DatasetStore,
    adapters_dir: Path,
    serving_dir: Path,
    shard: int,
    cfg: TrainConfig,
    log: Any = None,
) -> dict[str, Any]:
    """Retrain ``shard`` from the base model on its remaining examples and recompose serving.

    Precondition: the subject's examples were already dropped from the manifest (the dataset
    store's reclaim). The adapter store calls this.
    """
    t0 = time.time()
    out = adapters_dir / f"shard-{shard:02d}"
    before = adapter_hash(out)
    texts = [e.text for e in dataset.shard_examples(shard)]
    import shutil

    if out.exists():
        shutil.rmtree(out)
    if texts:
        meta = train_lora(texts, out, cfg, log=log)
    else:
        out.mkdir(parents=True, exist_ok=True)
        (out / "EMPTY").write_text("no examples in this shard\n")
        meta = {"examples": 0, "adapter_hash": ""}
    comp = compose_serving(adapters_dir, serving_dir, cfg.base_model)
    return {
        "method": "exact",
        "shard": shard,
        "adapter_hash_before": before,
        "adapter_hash_after": meta.get("adapter_hash", ""),
        "shard_examples_after": len(texts),
        "serving": comp,
        "wall_clock_s": round(time.time() - t0, 2),
    }


# --- approximate --------------------------------------------------------------------------------


def _encode(tok: Any, texts: Sequence[str], max_len: int) -> dict[str, Any]:
    enc = tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels
    return {k: v.to(device()) for k, v in enc.items()}


def _seq_logprob(model: Any, batch: dict[str, Any]) -> Any:
    """Per-sequence summed log-prob of the labels (differentiable)."""
    torch = require_torch()
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    mask = (labels != -100).float()
    safe = labels.clamp(min=0)
    lp = torch.log_softmax(logits, dim=-1).gather(2, safe.unsqueeze(2)).squeeze(2)
    return (lp * mask).sum(dim=1)


def approximate_unlearn(
    base_model: str,
    adapter_dir: Path,
    out_dir: Path,
    forget_texts: Sequence[str],
    retain_texts: Sequence[str],
    cfg: UnlearnConfig,
    log: Any = None,
) -> dict[str, Any]:
    """NPO or gradient-difference on a single adapter; writes the updated adapter to out_dir."""
    torch = require_torch()
    torch.manual_seed(cfg.seed)
    tok, pm = load_adapter_model(base_model, adapter_dir)
    pm.train()
    params = [p for p in pm.parameters() if p.requires_grad]
    if not params:
        # loaded for inference: make the LoRA weights trainable again
        for n, p in pm.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
        params = [p for p in pm.parameters() if p.requires_grad]
    ref = None
    if cfg.method == "npo":
        ref = copy.deepcopy(pm).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    import random

    rng = random.Random(cfg.seed)
    retain = list(retain_texts)[: cfg.retain_size] or list(forget_texts)[:1]
    forget = list(forget_texts)
    t0 = time.time()
    history: list[dict[str, float]] = []
    for step in range(cfg.steps):
        fb = _encode(tok, rng.sample(forget, min(cfg.batch_size, len(forget))), cfg.max_len)
        rb = _encode(tok, rng.sample(retain, min(cfg.batch_size, len(retain))), cfg.max_len)
        if cfg.method == "npo":
            assert ref is not None
            lp = _seq_logprob(pm, fb)
            with torch.no_grad():
                lp_ref = _seq_logprob(ref, fb)
            ratio = lp - lp_ref
            forget_loss = (
                -(2.0 / cfg.beta) * torch.nn.functional.logsigmoid(-cfg.beta * ratio).mean()
            )
        elif cfg.method == "gradient_difference":
            # gradient *ascent* on the forget NLL: NLL = -logprob, so the loss to minimise is
            # +logprob (normalised per token so β-free and comparable to the retain term)
            forget_loss = _seq_logprob(pm, fb).mean() / max(1, fb["labels"].shape[1])
        else:
            raise ValueError(f"unknown method {cfg.method!r}")
        retain_loss = -_seq_logprob(pm, rb).mean() / max(1, rb["labels"].shape[1])
        loss = forget_loss + cfg.retain_weight * retain_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        history.append(
            {
                "step": step,
                "forget_loss": float(forget_loss.item()),
                "retain_loss": float(retain_loss.item()),
            }
        )
        if log is not None and (step % 10 == 0 or step == cfg.steps - 1):
            log(
                f"{cfg.method} step {step + 1}/{cfg.steps} forget {float(forget_loss.item()):.3f} retain {float(retain_loss.item()):.3f}"
            )
    import shutil

    if out_dir.exists() and out_dir != adapter_dir:
        shutil.rmtree(out_dir)
    pm.save_pretrained(str(out_dir))
    meta = {
        "method": cfg.method,
        "config": cfg.to_dict(),
        "forget_examples": len(forget),
        "retain_examples": len(retain),
        "history": history,
        "wall_clock_s": round(time.time() - t0, 2),
        "adapter_hash": adapter_hash(out_dir),
    }
    (out_dir / "tombstone.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    return meta


def relearn(
    base_model: str,
    adapter_dir: Path,
    out_dir: Path,
    unrelated_texts: Sequence[str],
    steps: int,
    lr: float = 1e-4,
    batch_size: int = 4,
    max_len: int = 96,
    seed: int = 1,
) -> dict[str, Any]:
    """The relearning attack (Phase 7.6): light continued training on unrelated data."""
    torch = require_torch()
    torch.manual_seed(seed)
    tok, pm = load_adapter_model(base_model, adapter_dir)
    pm.train()
    for n, p in pm.named_parameters():
        p.requires_grad_("lora_" in n)
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    import random

    rng = random.Random(seed)
    texts = list(unrelated_texts)
    for _ in range(steps):
        b = _encode(tok, rng.sample(texts, min(batch_size, len(texts))), max_len)
        loss = -_seq_logprob(pm, b).mean() / max(1, b["labels"].shape[1])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    import shutil

    if out_dir.exists() and out_dir != adapter_dir:
        shutil.rmtree(out_dir)
    pm.save_pretrained(str(out_dir))
    return {"steps": steps, "adapter_hash": adapter_hash(out_dir)}
