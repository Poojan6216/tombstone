"""Lazy torch/transformers/peft access shared by the training modules ([train] extra)."""

from __future__ import annotations

import os
from typing import Any

from tombstone.errors import ConfigError


def require_torch() -> Any:
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ConfigError(
            "the model leg needs the [train] extra: uv pip install 'tombstone-erase[train]'"
        ) from e
    threads = int(os.environ.get("TOMBSTONE_TORCH_THREADS", "0") or 0)
    if threads > 0:
        torch.set_num_threads(threads)
    return torch


def load_base(model_name: str, dtype: str = "float32") -> tuple[Any, Any]:
    """(tokenizer, model) for a causal LM on CPU/GPU. Cached by transformers."""
    torch = require_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok: Any = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype))
    model.to(device())
    return tok, model


def device() -> str:
    torch = require_torch()
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
