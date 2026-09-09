"""8.1: the ten-minute path, scripted, plus the eight most likely mistakes with actionable messages."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.conftest import requires_faiss, requires_langchain
from tombstone.cli import main
from tombstone.errors import ConfigError, ConfirmationRequired, LineageGapError, SagaError

pytestmark = [requires_langchain, requires_faiss]


def _walkthrough(root: Path, capsys: pytest.CaptureFixture[str]) -> float:
    from langchain_core.documents import Document
    from langchain_core.indexing import InMemoryRecordManager, index

    from tombstone.integrations.langchain import TombstoneVectorStore
    from tombstone.lineage.stamp import stamp
    from tombstone.registry import Runtime

    t0 = time.perf_counter()
    # 1. init
    assert main(["init", "--path", str(root)]) == 0
    cfg = root / "tombstone.yaml"
    cfg.write_text(
        cfg.read_text().replace(
            "stores:\n",
            'stores:\n  - { name: "faiss:kb", kind: faiss, path: ./kb.index, embedding: hash-embed-64 }\n',
        )
    )
    # 2. wrap the vector store in one line
    vs = TombstoneVectorStore.from_config("faiss:kb", config=cfg)
    # 3. stamp at ingest (a stock LangChain flow: documents → index())
    pepper = Runtime.shared(cfg).pepper()
    docs = [
        stamp(
            Document(
                page_content=f"Customer {i} wrote about order {i}. Membership {i}.",
                metadata={"source": f"f{i}.txt"},
            ),
            f"S-{i}",
            f"f{i}.txt",
            "default",
            pepper=pepper,
        )
        for i in range(20)
    ]
    rm = InMemoryRecordManager("app")
    rm.create_schema()
    index(docs, rm, vs, cleanup="incremental", source_id_key="source")
    # 4. trace
    assert main(["trace", "--config", str(cfg), "--subject", "S-3", "--json"]) == 0
    trace_id = __import__("json").loads(capsys.readouterr().out)["trace_id"]
    # 5. erase
    assert (
        main(["erase", "--config", str(cfg), "--trace", trace_id, "--reason", "dsr-1", "--confirm"])
        == 0
    )
    capsys.readouterr()
    # 6. receipt
    assert main(["receipt", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "VERIFIED" in out and "not a legal instrument" in out
    Runtime.shared(cfg).close()
    return time.perf_counter() - t0


def test_ten_minute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    seconds = _walkthrough(tmp_path, capsys)
    assert seconds < 600, f"walkthrough took {seconds:.0f}s"


def _init(tmp_path: Path) -> Path:
    main(["init", "--path", str(tmp_path)])
    cfg = tmp_path / "tombstone.yaml"
    cfg.write_text(
        cfg.read_text().replace(
            "stores:\n",
            'stores:\n  - { name: "faiss:kb", kind: faiss, path: ./kb.index, embedding: hash-embed-64 }\n',
        )
    )
    return cfg


# --- the eight most likely mistakes --------------------------------------------------------------


def test_mistake_1_no_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "nowhere"))
    with pytest.raises(ConfigError, match="tombstone init"):
        from tombstone.registry import Runtime

        Runtime.load(None)


def test_mistake_2_unstamped_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_core.documents import Document

    from tombstone.integrations.langchain import TombstoneVectorStore

    monkeypatch.chdir(tmp_path)
    cfg = _init(tmp_path)
    vs = TombstoneVectorStore.from_config("faiss:kb", config=cfg)
    with pytest.raises(ValueError, match="stamp\\(doc, subject_id, source_id, scope"):
        vs.add_documents([Document(page_content="x", metadata={"source": "a"})])


def test_mistake_3_stamp_without_pepper() -> None:
    from tombstone.lineage.stamp import stamp

    with pytest.raises(ValueError, match="pepper is required"):
        stamp({}, "S-1", "a.txt", "default")


def test_mistake_4_erase_without_confirm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _init(tmp_path)
    rc = main(["erase", "--config", str(cfg), "--trace", "01X", "--reason", "r"])
    assert rc == ConfirmationRequired.exit_code
    assert "--confirm" in capsys.readouterr().err


def test_mistake_5_erase_by_subject_instead_of_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _init(tmp_path)
    rc = main(["erase", "--config", str(cfg), "--trace", "S-0417", "--reason", "r", "--confirm"])
    assert rc == SagaError.exit_code
    assert "trace id" in capsys.readouterr().err


def test_mistake_6_trace_unknown_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _init(tmp_path)
    rc = main(["trace", "--config", str(cfg), "--subject", "nobody"])
    assert rc == LineageGapError.exit_code
    assert "no lineage records" in capsys.readouterr().err


def test_mistake_7_config_typo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _init(tmp_path)
    cfg.write_text(cfg.read_text().replace("out_of_scope:", "out_of_scpoe:"))
    rc = main(["status", "--config", str(cfg)])
    err = capsys.readouterr().err
    assert rc == 1 and "out_of_scope" in err and "line" in err


def test_mistake_8_missing_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _init(tmp_path)
    cfg.write_text(
        cfg.read_text().replace(
            "kind: faiss, path: ./kb.index", "kind: qdrant, url: http://127.0.0.1:1"
        )
    )
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):  # noqa: ANN001, ANN202
        if name.startswith("qdrant_client"):
            raise ImportError("No module named 'qdrant_client'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    rc = main(["status", "--config", str(cfg)])
    err = capsys.readouterr().err
    assert rc == 1 and "tombstone-erase[qdrant]" in err
