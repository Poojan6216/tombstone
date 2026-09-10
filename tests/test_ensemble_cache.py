"""The cached ensemble decode must equal the uncached reference, token for token.

``ShardEnsemble.generate_greedy`` keeps a KV cache per shard and accumulates each shard's
context log-likelihood as a running sum. That is only worth doing if it is *exactly* the same
computation as re-scoring the whole sequence every step, because the unlearning headline
(canary extraction under the serving composition) is read off this decode.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.train, pytest.mark.slow]

MODEL = os.environ.get("TOMBSTONE_TEST_MODEL", "Qwen/Qwen2.5-0.5B")
ADAPTERS = Path(__file__).resolve().parents[1] / "bench" / "_work" / "unlearn" / "adapters"


def test_cached_decode_matches_the_uncached_reference() -> None:
    from tombstone.train.ensemble import ShardEnsemble

    shards = sorted(
        d.name for d in ADAPTERS.glob("shard-*") if (d / "adapter_config.json").is_file()
    )
    if len(shards) < 2:
        pytest.skip("needs at least two shard adapters (run bench/unlearn/run_unlearn.py)")
    shards = shards[:3]  # three shards is enough to exercise the gating; loading 16 is not
    ens = ShardEnsemble(MODEL, ADAPTERS, shards)
    prompts = [
        "The account holder on the shared address was",
        "Case note. The agent linked this case to the household of",
        "A satisfaction survey was sent after",
    ]
    for prefix in prompts:
        fast = ens.generate_greedy(prefix, max_new_tokens=12)
        slow = ens.generate_greedy_uncached(prefix, max_new_tokens=12)
        assert fast == slow, f"cached decode diverged on {prefix!r}: {fast!r} != {slow!r}"
