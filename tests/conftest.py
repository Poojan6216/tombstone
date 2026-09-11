from __future__ import annotations

import os
import secrets
import uuid
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

from tests import _pg

pytest_plugins: list[str] = []


def pytest_configure(config: pytest.Config) -> None:
    # Intel macOS: torch ships Intel's libiomp5 and faiss-cpu ships LLVM's libomp. Two OpenMP
    # runtimes in one process crash the moment either spawns worker threads (a libomp worker
    # blocks on a libiomp5 barrier — SIGSEGV). Keep both at one thread for the whole session.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # Keep every test's logs out of the user's terminal and away from stdout.
    os.environ.setdefault("TOMBSTONE_LOG_LEVEL", "warning")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chromadb
    os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")


@pytest.fixture
def pepper() -> bytes:
    return secrets.token_bytes(32)


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    from tombstone.config import Installation

    inst = Installation(tmp_path / ".tombstone")
    inst.ensure()
    return inst.root


@pytest.fixture(scope="session")
def pg() -> Iterator[_pg.PgHandle]:
    handle = _pg.acquire()
    if handle is None:
        pytest.skip("no Postgres available (set TOMBSTONE_PG_DSN, install postgresql, or pgserver)")
    yield handle
    handle.stop()


def _with_database(dsn: str, name: str) -> str:
    """Point a DSN at a different database, changing nothing else.

    String surgery is not safe here. A server reached over a unix socket carries its socket
    directory as a query parameter — ``postgresql://postgres@/postgres?host=/tmp/pg-xyz`` — and
    splitting on the last "/" lands inside that path, so the database name is appended to the
    *host* instead of replacing the database. The connection then looks for a socket in a
    directory that does not exist, which reads as "is the server running?" and sends you hunting
    for a server that is running perfectly well.
    """
    parts = urlsplit(dsn)
    return urlunsplit(parts._replace(path=f"/{name}"))


@pytest.fixture
def pg_database(pg: _pg.PgHandle) -> Iterator[str]:
    """A fresh database per test, dropped afterwards."""
    import psycopg

    name = f"tomb_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(pg.dsn, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    dsn = _with_database(pg.dsn, name)
    with psycopg.connect(dsn, autocommit=True) as conn:
        if pg.has_vector:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if pg.has_pgstattuple:
            conn.execute("CREATE EXTENSION IF NOT EXISTS pgstattuple")
    yield dsn
    with psycopg.connect(pg.dsn, autocommit=True) as conn:
        conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", (name,)
        )
        conn.execute(f'DROP DATABASE IF EXISTS "{name}"')


def has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


requires_chroma = pytest.mark.skipif(not has_module("chromadb"), reason="chromadb not installed")
requires_faiss = pytest.mark.skipif(not has_module("faiss"), reason="faiss not installed")
requires_qdrant = pytest.mark.skipif(
    not has_module("qdrant_client"), reason="qdrant-client not installed"
)
requires_langchain = pytest.mark.skipif(
    not has_module("langchain_core"), reason="langchain not installed"
)
requires_torch = pytest.mark.skipif(not has_module("torch"), reason="torch not installed")
requires_mcp = pytest.mark.skipif(not has_module("mcp"), reason="mcp not installed")
requires_st = pytest.mark.skipif(
    not has_module("sentence_transformers"), reason="sentence-transformers not installed"
)
