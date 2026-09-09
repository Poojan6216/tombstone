from __future__ import annotations

import os
import secrets
from pathlib import Path

import pytest

from tombstone.config import ConfigError, Installation
from tombstone.model.artifacts import SubjectRef


def test_same_raw_same_hmac(pepper: bytes) -> None:
    a = SubjectRef.from_raw("S-0417", pepper)
    b = SubjectRef.from_raw("S-0417", pepper)
    assert a == b
    assert len(a.hmac) == 64
    assert "S-0417" not in a.hmac and "S-0417" not in str(a) and "S-0417" not in a.short


def test_different_pepper_different_hmac() -> None:
    a = SubjectRef.from_raw("S-0417", secrets.token_bytes(32))
    b = SubjectRef.from_raw("S-0417", secrets.token_bytes(32))
    assert a != b


def test_different_raw_different_hmac(pepper: bytes) -> None:
    assert SubjectRef.from_raw("S-0417", pepper) != SubjectRef.from_raw("S-0418", pepper)


def test_empty_raw_rejected(pepper: bytes) -> None:
    with pytest.raises(ValueError):
        SubjectRef.from_raw("", pepper)


def test_short_pepper_rejected() -> None:
    with pytest.raises(ValueError):
        SubjectRef.from_raw("S-1", b"short")


def test_unicode_subject_ids(pepper: bytes) -> None:
    a = SubjectRef.from_raw("Zoë Ångström 李雷", pepper)
    assert len(a.hmac) == 64


def test_pepper_created_0600_and_gitignored(tmp_path: Path) -> None:
    inst = Installation(tmp_path / ".tombstone")
    inst.ensure()
    mode = inst.pepper_path.stat().st_mode & 0o777
    assert mode == 0o600
    assert len(inst.read_pepper()) == 32
    assert (inst.root / ".gitignore").read_text() == "*\n"
    assert inst.private_key_path.stat().st_mode & 0o777 == 0o600
    # idempotent: a second ensure() keeps the same pepper
    before = inst.read_pepper()
    inst.ensure()
    assert inst.read_pepper() == before


def test_pepper_bad_mode_rejected(tmp_path: Path) -> None:
    inst = Installation(tmp_path / ".tombstone")
    inst.ensure()
    os.chmod(inst.pepper_path, 0o644)
    with pytest.raises(ConfigError, match="0600"):
        inst.read_pepper()


def test_missing_pepper_is_actionable(tmp_path: Path) -> None:
    inst = Installation(tmp_path / ".tombstone")
    with pytest.raises(ConfigError, match="tombstone init"):
        inst.read_pepper()
