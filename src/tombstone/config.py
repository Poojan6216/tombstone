"""Configuration: ``./tombstone.yaml``, then ``$XDG_CONFIG_HOME/tombstone/config.yaml``.

Pydantic v2, schema-versioned. Every validation error names the field and the YAML line.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from tombstone.errors import ConfigError

CONFIG_SCHEMA_VERSION = 1
CONFIG_FILENAME = "tombstone.yaml"
STATE_DIR = ".tombstone"

_STORE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,99}$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LineageConfig(_Strict):
    backend: Literal["sqlite", "postgres"] = "sqlite"
    path: str = "./.tombstone/lineage.db"
    dsn: str | None = None

    @model_validator(mode="after")
    def _check_dsn(self) -> LineageConfig:
        if self.backend == "postgres" and not self.dsn:
            raise ValueError("lineage.dsn is required when lineage.backend is 'postgres'")
        return self


StoreKind = Literal[
    "chroma",
    "pgvector",
    "qdrant",
    "faiss",
    "cache_exact",
    "cache_semantic",
    "dataset",
    "adapter",
    "docstore",
    "memory",
]


class StoreConfig(_Strict):
    """One store entry. Backend-specific keys are validated per ``kind``."""

    name: str
    kind: StoreKind
    # vector backends
    path: str | None = None
    dsn: str | None = None
    table: str | None = None
    collection: str | None = None
    url: str | None = None
    embedding: str | None = None
    # caches
    backing: str | None = None
    # dataset / adapter
    manifest: str | None = None
    shards: int | None = None
    base_model: str | None = None
    # any backend
    scope: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _STORE_NAME_RE.match(v):
            raise ValueError(
                f"store name {v!r} is invalid; use letters, digits, '.', '_', ':', '/', '-'"
            )
        return v

    @model_validator(mode="after")
    def _per_kind(self) -> StoreConfig:
        k = self.kind
        need: dict[str, list[str]] = {
            "chroma": ["path"],
            "faiss": ["path"],
            "qdrant": ["path|url"],
            "pgvector": ["dsn", "table"],
            "cache_exact": ["path"],
            "cache_semantic": ["backing"],
            "dataset": ["manifest"],
            "adapter": ["path"],
            "docstore": ["path"],
            "memory": [],
        }
        for spec in need[k]:
            alternatives = spec.split("|")
            if not any(getattr(self, a) for a in alternatives):
                raise ValueError(
                    f"store {self.name!r} of kind {k!r} requires "
                    + " or ".join(f"'{a}'" for a in alternatives)
                )
        if k in {"chroma", "pgvector", "qdrant", "faiss"} and not self.embedding:
            raise ValueError(
                f"store {self.name!r} of kind {k!r} requires 'embedding' (the embedding model "
                "name, e.g. all-MiniLM-L6-v2) so lineage edges can be tagged with it"
            )
        if k == "adapter" and self.shards is not None and self.shards < 1:
            raise ValueError(f"store {self.name!r}: shards must be >= 1")
        return self


class EraseConfig(_Strict):
    require_confirm: bool = True
    reclaim_timeout_s: Annotated[int, Field(ge=1)] = 600
    semantic_probe_budget: Annotated[int, Field(ge=1, le=100)] = 5
    logical_probe_k: Annotated[int, Field(ge=1, le=1000)] = 40
    lock_timeout_s: Annotated[float, Field(gt=0)] = 30.0


class ModelConfig(_Strict):
    base: str = "Qwen/Qwen2.5-0.5B"
    unlearn: Literal["exact", "npo", "gradient_difference"] = "exact"
    mia_reference: str | None = None
    unlearn_steps: Annotated[int, Field(ge=1)] = 40
    unlearn_lr: Annotated[float, Field(gt=0)] = 1e-4


class TombstoneConfig(_Strict):
    version: int
    scope: str = "default"
    lineage: LineageConfig = Field(default_factory=LineageConfig)
    stores: list[StoreConfig] = Field(default_factory=list)
    erase: EraseConfig = Field(default_factory=EraseConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    out_of_scope: list[str]
    state_dir: str = "./.tombstone"

    @field_validator("version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported config schema version {v}; this build understands "
                f"version {CONFIG_SCHEMA_VERSION}"
            )
        return v

    @field_validator("scope")
    @classmethod
    def _scope(cls, v: str) -> str:
        if not v:
            raise ValueError("scope must be a non-empty tenant name ('default' if single-tenant)")
        return v

    @field_validator("out_of_scope")
    @classmethod
    def _oos(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        if not cleaned:
            raise ValueError(
                "out_of_scope must list at least one layer this tool cannot reach (for example "
                "'database backups and snapshots', 'write-ahead logs and replicas', "
                "'embedding-provider-side request logs'). It is required because a receipt that "
                "claims to have reached every layer would be false: every receipt repeats this "
                "list so the reader knows what was not checked (Hard Rule 2)."
            )
        return cleaned

    @field_validator("stores")
    @classmethod
    def _stores(cls, stores: list[StoreConfig]) -> list[StoreConfig]:
        names = [s.name for s in stores]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate store names: {', '.join(dupes)}")
        for s in stores:
            if s.kind == "cache_semantic" and s.backing not in names:
                raise ValueError(
                    f"store {s.name!r}: backing {s.backing!r} is not a configured store"
                )
        return stores

    def store(self, name: str) -> StoreConfig:
        for s in self.stores:
            if s.name == name:
                return s
        raise ConfigError(
            f"no store named {name!r} in config; configured: {[s.name for s in self.stores]}"
        )

    def dump(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json", exclude_none=True), sort_keys=False)


def resolve_env(value: str | None) -> str | None:
    """``env:NAME`` → ``os.environ[NAME]``; anything else returned verbatim."""
    if value is None:
        return None
    if value.startswith("env:"):
        name = value[4:]
        if name not in os.environ:
            raise ConfigError(f"environment variable {name!r} referenced by config is not set")
        return os.environ[name]
    return value


# --- loading with line numbers -----------------------------------------------------------------


class _LineLoader(yaml.SafeLoader):
    """A SafeLoader that records the source line of every mapping key as ``__line__<key>``."""


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    lines: dict[str, int] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        value = loader.construct_object(value_node, deep=True)
        mapping[key] = value
        lines[str(key)] = key_node.start_mark.line + 1
    mapping["__lines__"] = lines
    return mapping


_LineLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _strip_lines(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_lines(v) for k, v in obj.items() if k != "__lines__"}
    if isinstance(obj, list):
        return [_strip_lines(v) for v in obj]
    return obj


def _line_for(raw: Any, loc: tuple[Any, ...]) -> int | None:
    """Walk ``loc`` through the line-annotated tree and return the closest known line."""
    node = raw
    best: int | None = None
    for part in loc:
        if isinstance(node, dict):
            lines = node.get("__lines__", {})
            if str(part) in lines:
                best = lines[str(part)]
            node = node.get(part)
        elif isinstance(node, list) and isinstance(part, int) and 0 <= part < len(node):
            node = node[part]
            if isinstance(node, dict) and node.get("__lines__"):
                best = min(int(v) for v in node["__lines__"].values())
        else:
            break
    return best


def _format_errors(err: ValidationError, raw: Any, path: Path | None) -> str:
    lines: list[str] = []
    where = f" in {path}" if path else ""
    for e in err.errors():
        loc = e["loc"]
        field = ".".join(str(p) for p in loc) or "<root>"
        line = _line_for(raw, loc)
        at = f" (line {line})" if line else ""
        msg = e["msg"]
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        if e["type"] == "missing":
            msg = f"required field is missing: {field}"
        elif e["type"] == "extra_forbidden":
            msg = f"unknown field {field!r} (typo? check tombstone.yaml.example)"
        lines.append(f"config error{where}: {field}{at}: {msg}")
    return "\n".join(lines)


def parse_config(text: str, path: Path | None = None) -> TombstoneConfig:
    try:
        raw = yaml.load(text, Loader=_LineLoader)  # noqa: S506 - _LineLoader is a SafeLoader
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        line = f" (line {mark.line + 1})" if mark else ""
        raise ConfigError(
            f"config error{f' in {path}' if path else ''}: invalid YAML{line}: {e}"
        ) from e
    if raw is None:
        raise ConfigError(f"config error{f' in {path}' if path else ''}: file is empty")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config error{f' in {path}' if path else ''}: top level must be a mapping, "
            f"got {type(_strip_lines(raw)).__name__}"
        )
    clean = _strip_lines(raw)
    if "version" not in clean:
        raise ConfigError(
            f"config error{f' in {path}' if path else ''}: version (line 1): required field is "
            f"missing: version — add `version: {CONFIG_SCHEMA_VERSION}` at the top"
        )
    try:
        return TombstoneConfig.model_validate(clean)
    except ValidationError as err:
        raise ConfigError(_format_errors(err, raw, path)) from err


def config_search_paths(cwd: Path | None = None) -> list[Path]:
    cwd = cwd or Path.cwd()
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return [cwd / CONFIG_FILENAME, Path(xdg) / "tombstone" / "config.yaml"]


def load_config(explicit: str | os.PathLike[str] | None = None) -> tuple[TombstoneConfig, Path]:
    candidates = [Path(explicit)] if explicit else config_search_paths()
    for p in candidates:
        if p.is_file():
            return parse_config(p.read_text(encoding="utf-8"), p), p
    tried = ", ".join(str(p) for p in candidates)
    raise ConfigError(f"no config found (tried: {tried}). Run `tombstone init` first.")


class Installation:
    """Paths inside the state dir. Created by ``tombstone init``; never committed."""

    def __init__(self, root: str | os.PathLike[str] = STATE_DIR) -> None:
        self.root = Path(root)

    @property
    def pepper_path(self) -> Path:
        return self.root / "pepper"

    @property
    def keys_dir(self) -> Path:
        return self.root / "keys"

    @property
    def private_key_path(self) -> Path:
        return self.keys_dir / "ed25519.key"

    @property
    def public_key_path(self) -> Path:
        return self.keys_dir / "ed25519.pub"

    @property
    def ledger_path(self) -> Path:
        return self.root / "ledger.jsonl"

    @property
    def journal_path(self) -> Path:
        return self.root / "journal.jsonl"

    @property
    def dlq_path(self) -> Path:
        return self.root / "dlq.jsonl"

    @property
    def receipts_dir(self) -> Path:
        return self.root / "receipts"

    @property
    def traces_dir(self) -> Path:
        return self.root / "traces"

    @property
    def lineage_db_path(self) -> Path:
        return self.root / "lineage.db"

    @property
    def probes_dir(self) -> Path:
        return self.root / "probes"

    def exists(self) -> bool:
        return self.pepper_path.is_file()

    def read_pepper(self) -> bytes:
        if not self.pepper_path.is_file():
            raise ConfigError(
                f"pepper not found at {self.pepper_path}; run `tombstone init` in this directory"
            )
        mode = self.pepper_path.stat().st_mode & 0o777
        if mode & 0o077:
            raise ConfigError(
                f"pepper at {self.pepper_path} has mode {oct(mode)}; it must be 0600. "
                f"Fix with: chmod 600 {self.pepper_path}"
            )
        data = self.pepper_path.read_bytes()
        if len(data) < 32:
            raise ConfigError(f"pepper at {self.pepper_path} is too short ({len(data)} bytes)")
        return data

    def ensure(self) -> None:
        """Create the state dir, pepper and key material if absent. Idempotent."""
        import secrets

        from tombstone.util import secure_write

        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ".gitignore").write_text("*\n", encoding="utf-8")
        if not self.pepper_path.is_file():
            secure_write(self.pepper_path, secrets.token_bytes(32), 0o600)
        self.keys_dir.mkdir(exist_ok=True, mode=0o700)
        if not self.private_key_path.is_file():
            from tombstone.receipt.sign import generate_keypair

            generate_keypair(self.private_key_path, self.public_key_path)
        for d in (self.receipts_dir, self.traces_dir, self.probes_dir):
            d.mkdir(exist_ok=True)
        for f in (self.ledger_path, self.journal_path, self.dlq_path):
            if not f.is_file():
                f.touch()
