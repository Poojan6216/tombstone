from __future__ import annotations

import json
from pathlib import Path

import pytest

from tombstone.cli import main
from tombstone.commands.init import detect_project, run_init
from tombstone.config import load_config


def _make_langchain_project(root: Path) -> None:
    (root / "app.py").write_text(
        "from langchain_core.documents import Document\nfrom langchain_core.indexing import index, RecordManager\n"
    )
    (root / "chroma").mkdir()
    (root / "chroma" / "chroma.sqlite3").write_bytes(b"SQLite format 3\x00")


def _make_faiss_project(root: Path) -> None:
    (root / "kb.index").write_bytes(b"IxMp")
    (root / "ingest.py").write_text("import langchain\n")


def _make_adapter_project(root: Path) -> None:
    (root / "adapters" / "support-v3").mkdir(parents=True)
    (root / "adapters" / "support-v3" / "adapter_config.json").write_text("{}")
    (root / ".langchain.db").write_bytes(b"")


def _make_empty_project(root: Path) -> None:
    (root / "README.md").write_text("nothing here\n")


SHAPES = {
    "langchain+chroma": _make_langchain_project,
    "faiss": _make_faiss_project,
    "adapter+cache": _make_adapter_project,
    "empty": _make_empty_project,
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_init_round_trip_over_project_shapes(
    tmp_path: Path, shape: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PG_DSN", "postgresql://x" if shape == "faiss" else "")
    if shape != "faiss":
        monkeypatch.delenv("PG_DSN", raising=False)
    SHAPES[shape](tmp_path)
    result = run_init(tmp_path)
    assert result["written"] is True
    cfg, path = load_config(tmp_path / "tombstone.yaml")
    assert path.is_file()
    names = [s.name for s in cfg.stores]
    det = detect_project(tmp_path)
    if shape == "langchain+chroma":
        assert det.langchain and det.record_manager
        assert any(n.startswith("chroma:") for n in names)
    if shape == "faiss":
        assert any(n.startswith("faiss:") for n in names)
        assert any(n.startswith("pgvector:") for n in names)
    if shape == "adapter+cache":
        assert any(n.startswith("lora/") for n in names)
        assert "exact-cache" in names
    assert cfg.out_of_scope  # always present
    state = tmp_path / ".tombstone"
    assert (state / "pepper").stat().st_mode & 0o777 == 0o600
    assert (state / "keys" / "ed25519.key").is_file()
    assert (state / "ledger.jsonl").is_file()
    assert (state / "lineage.db").is_file()
    assert ".tombstone/" in (tmp_path / ".gitignore").read_text()
    # dump → load is stable
    assert load_config(tmp_path / "tombstone.yaml")[0].dump() == cfg.dump()


def test_init_never_overwrites_without_backup(tmp_path: Path) -> None:
    _make_empty_project(tmp_path)
    run_init(tmp_path)
    cfg_path = tmp_path / "tombstone.yaml"
    original = cfg_path.read_bytes()
    custom = original + b"# my edit\n"
    cfg_path.write_bytes(custom)
    # without --force the existing config is kept as-is
    r = run_init(tmp_path)
    assert r["written"] is False and cfg_path.read_bytes() == custom
    # with --force a byte-for-byte backup exists before the write
    r = run_init(tmp_path, force=True)
    assert r["written"] is True and r["backup"]
    assert Path(str(r["backup"])).read_bytes() == custom
    assert cfg_path.read_bytes() != custom


def test_init_via_cli_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _make_empty_project(tmp_path)
    rc = main(["init", "--path", str(tmp_path), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["written"] is True
    assert "detected" in data


def test_init_keeps_pepper_across_runs(tmp_path: Path) -> None:
    _make_empty_project(tmp_path)
    run_init(tmp_path)
    pepper = (tmp_path / ".tombstone" / "pepper").read_bytes()
    run_init(tmp_path, force=True)
    assert (tmp_path / ".tombstone" / "pepper").read_bytes() == pepper
