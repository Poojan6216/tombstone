from __future__ import annotations

import subprocess
from pathlib import Path

from tombstone.train.canaries import canaries_for, canary_for, token_recovered


def test_deterministic_and_unique() -> None:
    ids = [f"S-{i:04d}" for i in range(200)]
    a = canaries_for(ids, 7)
    b = canaries_for(ids, 7)
    assert a == b
    assert len({c.token for c in a}) == 200
    assert all(len(c.token.replace("-", "")) == 12 for c in a)
    assert canaries_for(ids, 8) != a


def test_prefix_and_recovery() -> None:
    c = canary_for("S-0417", 1)
    assert c.sentence.startswith(c.prefix)
    assert c.token not in c.prefix
    assert token_recovered(c.prefix + " " + c.token + " and more", c)
    assert not token_recovered("nothing here", c)


def test_no_canary_plaintext_committed() -> None:
    """Regenerate the fixture corpus canaries and grep the tracked tree for their tokens."""
    from tests._corpus import load_corpus

    tokens = {d.canary.token for d in load_corpus() if d.canary}
    assert tokens
    root = Path(__file__).resolve().parents[1]
    files = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    for f in files:
        p = root / f
        if not p.is_file():
            continue
        try:
            data = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for t in tokens:
            assert t not in data, f"canary {t} committed in plaintext in {f}"
