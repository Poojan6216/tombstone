from __future__ import annotations

from pathlib import Path

import pytest

from tombstone.commands.init import run_init
from tombstone.config import (
    CONFIG_SCHEMA_VERSION,
    ConfigError,
    TombstoneConfig,
    load_config,
    parse_config,
    resolve_env,
)
from tombstone.registry import Runtime

GOOD = """\
version: 1
scope: default
lineage: { backend: sqlite, path: ./.tombstone/lineage.db }
stores:
  - { name: "chroma:kb-v2", kind: chroma, path: ./chroma, embedding: all-MiniLM-L6-v2 }
  - { name: "semantic-cache", kind: cache_semantic, backing: "chroma:kb-v2" }
  - { name: "ft-dataset", kind: dataset, manifest: ./train/manifest.json }
erase:
  require_confirm: true
  reclaim_timeout_s: 600
  semantic_probe_budget: 5
model:
  base: Qwen/Qwen2.5-0.5B
  unlearn: exact
out_of_scope:
  - "database backups and snapshots"
  - "write-ahead logs and replicas"
"""


def test_good_config_parses() -> None:
    cfg = parse_config(GOOD)
    assert cfg.version == CONFIG_SCHEMA_VERSION
    assert [s.name for s in cfg.stores] == ["chroma:kb-v2", "semantic-cache", "ft-dataset"]
    assert cfg.out_of_scope == ["database backups and snapshots", "write-ahead logs and replicas"]


def test_round_trip_is_stable() -> None:
    cfg = parse_config(GOOD)
    dumped = cfg.dump()
    cfg2 = parse_config(dumped)
    assert cfg2 == cfg
    assert cfg2.dump() == dumped


# Each malformed config must produce a specific, actionable error naming field and line.
MALFORMED: list[tuple[str, str, list[str]]] = [
    (
        "missing version",
        GOOD.replace("version: 1\n", ""),
        ["version", "required"],
    ),
    (
        "wrong version",
        GOOD.replace("version: 1", "version: 7"),
        ["version", "line 1", "unsupported"],
    ),
    (
        "missing out_of_scope",
        GOOD.split("out_of_scope:")[0],
        ["out_of_scope", "required"],
    ),
    (
        "empty out_of_scope",
        GOOD.split("out_of_scope:")[0] + "out_of_scope: []\n",
        ["out_of_scope", "line 15", "cannot reach", "Hard Rule 2"],
    ),
    (
        "unknown top-level field",
        GOOD + "telemetry: true\n",
        ["telemetry", "line 18", "unknown field"],
    ),
    (
        "store missing embedding",
        GOOD.replace(", embedding: all-MiniLM-L6-v2", ""),
        ["stores.0", "line 5", "embedding"],
    ),
    (
        # this case used "pinecone" until pinecone became a real backend, which is exactly the
        # trap: pick a kind nobody will plausibly implement, or the test quietly stops testing
        # what it says it does
        "store of unknown kind",
        GOOD.replace("kind: chroma", "kind: notarealvectordb"),
        ["stores.0.kind", "line 5"],
    ),
    (
        "chroma without path",
        GOOD.replace("path: ./chroma, ", ""),
        ["stores.0", "line 5", "'path'"],
    ),
    (
        "semantic cache backing unknown store",
        GOOD.replace('backing: "chroma:kb-v2"', 'backing: "nope"'),
        ["stores", "line 4", "backing", "not a configured store"],
    ),
    (
        "duplicate store names",
        GOOD.replace('name: "ft-dataset"', 'name: "chroma:kb-v2"'),
        ["stores", "line 4", "duplicate store names", "chroma:kb-v2"],
    ),
    (
        "bad store name",
        GOOD.replace('name: "chroma:kb-v2"', 'name: "chroma kb"', 1).replace(
            'backing: "chroma:kb-v2"', 'backing: "chroma kb"'
        ),
        ["stores.0.name", "line 5", "invalid"],
    ),
    (
        "invalid YAML",
        GOOD.replace("erase:\n", "erase: [\n"),
        ["invalid YAML"],
    ),
    (
        "scope empty",
        GOOD.replace("scope: default", 'scope: ""'),
        ["scope", "line 2", "non-empty"],
    ),
    (
        "reclaim timeout zero",
        GOOD.replace("reclaim_timeout_s: 600", "reclaim_timeout_s: 0"),
        ["erase.reclaim_timeout_s", "line 10"],
    ),
    (
        "probe budget too large",
        GOOD.replace("semantic_probe_budget: 5", "semantic_probe_budget: 1000"),
        ["erase.semantic_probe_budget", "line 11"],
    ),
    (
        "unknown unlearn method",
        GOOD.replace("unlearn: exact", "unlearn: magic"),
        ["model.unlearn", "line 14"],
    ),
    (
        "postgres lineage without dsn",
        GOOD.replace(
            "lineage: { backend: sqlite, path: ./.tombstone/lineage.db }",
            "lineage: { backend: postgres }",
        ),
        ["lineage", "dsn"],
    ),
    (
        "top level not a mapping",
        "- just\n- a list\n",
        ["top level must be a mapping"],
    ),
    (
        "empty file",
        "",
        ["empty"],
    ),
    (
        "dataset without manifest",
        GOOD.replace("manifest: ./train/manifest.json", "path: ./train"),
        ["stores.2", "line 7", "'manifest'"],
    ),
]


@pytest.mark.parametrize("label,text,expect", MALFORMED, ids=[m[0] for m in MALFORMED])
def test_malformed_configs_name_field_and_line(label: str, text: str, expect: list[str]) -> None:
    with pytest.raises(ConfigError) as ei:
        parse_config(text, Path("tombstone.yaml"))
    msg = str(ei.value)
    for needle in expect:
        assert needle in msg, f"{label}: expected {needle!r} in error:\n{msg}"


def test_out_of_scope_message_explains_why() -> None:
    with pytest.raises(ConfigError) as ei:
        parse_config(GOOD.split("out_of_scope:")[0] + "out_of_scope: []\n")
    msg = str(ei.value)
    assert "backups" in msg and "receipt" in msg


def test_load_config_search_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    xdg = tmp_path / "xdg"
    (xdg / "tombstone").mkdir(parents=True)
    (xdg / "tombstone" / "config.yaml").write_text(GOOD.replace("scope: default", "scope: xdg"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    cfg, path = load_config()
    assert cfg.scope == "xdg" and path.parent.name == "tombstone"
    (tmp_path / "tombstone.yaml").write_text(GOOD)
    cfg, path = load_config()
    assert cfg.scope == "default" and path.name == "tombstone.yaml"


def test_load_config_missing_is_actionable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "nowhere"))
    with pytest.raises(ConfigError, match="tombstone init"):
        load_config()


def test_resolve_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PG_DSN", "postgresql://x")
    assert resolve_env("env:PG_DSN") == "postgresql://x"
    assert resolve_env("literal") == "literal"
    with pytest.raises(ConfigError, match="NOPE"):
        resolve_env("env:NOPE")


def test_store_lookup() -> None:
    cfg: TombstoneConfig = parse_config(GOOD)
    assert cfg.store("ft-dataset").kind == "dataset"
    with pytest.raises(ConfigError, match="no store named"):
        cfg.store("missing")


@pytest.mark.parametrize("backend", ["qdrant", "faiss", "chroma"])
def test_all_stores_opens_a_semantic_cache_beside_its_backing_index(
    backend: str, tmp_path: Path
) -> None:
    """``all_stores()`` must be able to open every configured store at once.

    A semantic cache backed by a vector index gets its own derived store, and for a local file
    backend that store cannot share the backing index's path: Qdrant's local mode takes an
    exclusive lock on its storage folder, so the second client raises "already accessed by
    another instance of Qdrant client". Only the saga path calls ``all_stores()`` (through the
    pin check), so this surfaced two backends into a benchmark rather than at the first test.
    """
    from tests import _stores

    _stores.skip_unless(backend)
    root = tmp_path / "proj"
    root.mkdir()
    run_init(root)
    cfg_path = root / "tombstone.yaml"
    kb = {
        "qdrant": '{ name: "kb", kind: qdrant, path: ./qd, collection: kb-v1, embedding: hash-embed-64 }',
        "faiss": '{ name: "kb", kind: faiss, path: ./kb.index, embedding: hash-embed-64 }',
        "chroma": '{ name: "kb", kind: chroma, path: ./chroma, collection: kb-v1, embedding: hash-embed-64 }',
    }[backend]
    cfg_path.write_text(
        cfg_path.read_text().replace(
            "stores:\n",
            f'stores:\n  - {kb}\n  - {{ name: "cache", kind: cache_semantic, backing: "kb" }}\n',
        )
    )
    rt = Runtime.load(cfg_path)
    try:
        stores = rt.all_stores()  # the call the saga makes before it does anything
        assert {"kb", "cache"} <= set(stores)
        assert rt.all_stores() is not None, "must be repeatable"
    finally:
        rt.close()
