"""LoRA fine-tuning, one adapter per shard, plus a merged serving composition and a single
unsharded adapter for the approximate-unlearning comparison.

Serving composition: **weighted average of the shard adapters' LoRA deltas** (``merge_adapters``
with ``combination_type="linear"`` and equal weights). Chosen over sequential merge because it
is order-independent, so recomposing after one shard is retrained is deterministic and cheap.
Documented in ``docs/unlearning.md``.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tombstone.train._torch import device, load_base, require_torch
from tombstone.train.dataset import DatasetStore, Example
from tombstone.util import sha256_hex

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-0.5B"
    rank: int = 16
    alpha: int = 32
    lr: float = 3e-4
    epochs: int = 12
    batch_size: int = 8
    max_len: int = 96
    repeats: int = 3  # each canary-bearing example is repeated to make memorisation real
    seed: int = 20260908
    targets: tuple[str, ...] = DEFAULT_TARGETS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def adapter_hash(path: Path) -> str:
    """Hash over adapter_config.json + the weights file, so a retrain changes the pin."""
    h = []
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        f = path / name
        if f.is_file():
            h.append(sha256_hex(f.read_bytes()))
    return sha256_hex("|".join(h)) if h else ""


def _peft_model(model: Any, cfg: TrainConfig) -> Any:
    from peft import LoraConfig, get_peft_model

    lc = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        target_modules=list(cfg.targets),
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lc)


def _batches(
    tok: Any, texts: Sequence[str], cfg: TrainConfig, torch: Any, shuffle_seed: int
) -> list[dict[str, Any]]:
    import random

    order = list(range(len(texts)))
    random.Random(shuffle_seed).shuffle(order)
    out = []
    for i in range(0, len(order), cfg.batch_size):
        chunk = [texts[j] for j in order[i : i + cfg.batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_len)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        out.append({k: v.to(device()) for k, v in enc.items()})
    return out


def train_lora(
    texts: Sequence[str],
    out_dir: Path,
    cfg: TrainConfig,
    base: tuple[Any, Any] | None = None,
    log: Any = None,
) -> dict[str, Any]:
    """Train one LoRA adapter on ``texts`` from the base model; save to ``out_dir``."""
    torch = require_torch()
    torch.manual_seed(cfg.seed)
    tok, model = base if base is not None else load_base(cfg.base_model)
    pm = _peft_model(model, cfg)
    pm.train()
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    expanded = list(texts) * max(1, cfg.repeats)
    t0 = time.time()
    steps = 0
    last_loss = float("nan")
    for epoch in range(cfg.epochs):
        for batch in _batches(tok, expanded, cfg, torch, cfg.seed + epoch):
            out = pm(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
            last_loss = float(out.loss.item())
        if log is not None:
            log(
                f"epoch {epoch + 1}/{cfg.epochs} loss {last_loss:.3f} steps {steps} t {time.time() - t0:.0f}s"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(str(out_dir))
    meta = {
        "config": cfg.to_dict(),
        "examples": len(texts),
        "steps": steps,
        "final_loss": last_loss,
        "wall_clock_s": round(time.time() - t0, 2),
        "device": device(),
        "adapter_hash": adapter_hash(out_dir),
    }
    (out_dir / "tombstone.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    # detach the adapter from the shared base so the next shard starts clean
    pm.unload() if hasattr(pm, "unload") else None
    return meta


def train_shards(
    dataset: DatasetStore,
    adapters_dir: Path,
    cfg: TrainConfig,
    shards: Sequence[int] | None = None,
    log: Any = None,
) -> dict[str, Any]:
    """One adapter per shard under ``adapters_dir/shard-NN``; returns per-shard metadata."""
    base = load_base(cfg.base_model)
    result: dict[str, Any] = {"shards": {}, "config": cfg.to_dict()}
    todo = list(shards) if shards is not None else list(range(dataset.shards()))
    for shard in todo:
        examples = dataset.shard_examples(shard)
        texts = [e.text for e in examples]
        out = adapters_dir / f"shard-{shard:02d}"
        if out.exists():
            shutil.rmtree(out)
        if not texts:
            out.mkdir(parents=True, exist_ok=True)
            (out / "EMPTY").write_text("no examples in this shard\n")
            result["shards"][str(shard)] = {"examples": 0, "adapter_hash": ""}
            continue
        if log is not None:
            log(f"shard {shard}: {len(texts)} examples")
        meta = train_lora(texts, out, cfg, base=base, log=log)
        meta["example_ids"] = [e.example_id for e in examples]
        result["shards"][str(shard)] = meta
    return result


def train_unsharded(
    dataset: DatasetStore, out_dir: Path, cfg: TrainConfig, log: Any = None
) -> dict[str, Any]:
    examples: list[Example] = []
    for shard in range(dataset.shards()):
        examples.extend(dataset.shard_examples(shard))
    meta = train_lora([e.text for e in examples], out_dir, cfg, log=log)
    meta["example_ids"] = [e.example_id for e in examples]
    return meta


def compose_serving(
    adapters_dir: Path, serving_dir: Path, base_model: str, exclude: Sequence[int] = ()
) -> dict[str, Any]:
    """Average the shard adapters (minus ``exclude``) into one serving adapter at ``serving_dir``."""
    from peft import PeftModel

    _tok, model = load_base(base_model)
    shard_dirs = sorted(
        p
        for p in adapters_dir.glob("shard-*")
        if p.is_dir()
        and (p / "adapter_config.json").is_file()
        and int(p.name.split("-")[1]) not in set(exclude)
    )
    if not shard_dirs:
        raise ValueError("no shard adapters to compose")
    pm = PeftModel.from_pretrained(model, str(shard_dirs[0]), adapter_name=shard_dirs[0].name)
    for d in shard_dirs[1:]:
        pm.load_adapter(str(d), adapter_name=d.name)
    names = [d.name for d in shard_dirs]
    weights = [1.0] * len(names)
    pm.add_weighted_adapter(names, weights, "serving", combination_type="linear")
    pm.set_adapter("serving")
    if serving_dir.exists():
        shutil.rmtree(serving_dir)
    pm.save_pretrained(str(serving_dir), selected_adapters=["serving"])
    # peft saves selected adapters into subdirectories; flatten so the path is a plain adapter
    sub = serving_dir / "serving"
    if sub.is_dir():
        for f in sub.iterdir():
            shutil.move(str(f), str(serving_dir / f.name))
        sub.rmdir()
    meta = {
        "composition": "linear average of shard adapters (equal weights)",
        "shards": names,
        "excluded": sorted(set(exclude)),
        "adapter_hash": adapter_hash(serving_dir),
    }
    (serving_dir / "tombstone.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    return meta


def load_adapter_model(base_model: str, adapter_dir: Path) -> tuple[Any, Any]:
    from peft import PeftModel

    tok, model = load_base(base_model)
    pm = PeftModel.from_pretrained(model, str(adapter_dir))
    pm.eval()
    return tok, pm
