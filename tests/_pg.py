"""Find or start a Postgres for tests, without Docker.

Order: ``TOMBSTONE_PG_DSN`` → a local server binary (Homebrew postgresql@16/17, PATH) with a
temporary cluster → ``pgserver`` (pip-installed Postgres+pgvector) → testcontainers (Docker) →
skip. The returned handle says which extensions are available so tests can assert on
``UNVERIFIED`` outcomes honestly rather than skipping them.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

_CANDIDATE_BIN_DIRS = [
    os.environ.get("TOMBSTONE_PG_BIN", ""),
    "/usr/local/opt/postgresql@16/bin",
    "/opt/homebrew/opt/postgresql@16/bin",
    "/usr/local/opt/postgresql@17/bin",
    "/opt/homebrew/opt/postgresql@17/bin",
    "/usr/lib/postgresql/16/bin",
    "/usr/lib/postgresql/17/bin",
]


@dataclass
class PgHandle:
    dsn: str
    kind: str  # "env" | "local" | "pgserver" | "docker"
    has_vector: bool = False
    has_pgstattuple: bool = False
    version: str = ""
    _cleanup: list[object] = field(default_factory=list)

    def stop(self) -> None:
        for c in self._cleanup:
            try:
                if callable(c):
                    c()
            except Exception:  # noqa: BLE001
                pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _probe(dsn: str) -> tuple[bool, bool, str]:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        ver = str(conn.execute("SHOW server_version").fetchone()[0])  # type: ignore[index]
        avail = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM pg_available_extensions WHERE name IN ('vector','pgstattuple')"
            ).fetchall()
        }
        has_vector = "vector" in avail
        has_stat = "pgstattuple" in avail
        if has_vector:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if has_stat:
            conn.execute("CREATE EXTENSION IF NOT EXISTS pgstattuple")
    return has_vector, has_stat, ver


def _start_local(bindir: Path, workdir: Path) -> PgHandle | None:
    initdb, pg_ctl = bindir / "initdb", bindir / "pg_ctl"
    if not (initdb.is_file() and pg_ctl.is_file()):
        return None
    data = workdir / "pgdata"
    port = _free_port()
    # macOS: without a valid locale the postmaster "becomes multithreaded during startup".
    env = {**os.environ, "LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}
    try:
        subprocess.run(
            [str(initdb), "-D", str(data), "-U", "postgres", "--auth=trust", "-E", "UTF8"],
            check=True,
            capture_output=True,
            timeout=120,
            env=env,
        )
        opts = f"-p {port} -k {workdir} -c listen_addresses=127.0.0.1 -c fsync=off"
        subprocess.run(
            [
                str(pg_ctl),
                "-D",
                str(data),
                "-o",
                opts,
                "-l",
                str(workdir / "pg.log"),
                "-w",
                "start",
            ],
            check=True,
            capture_output=True,
            timeout=120,
            env=env,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    dsn = f"postgresql://postgres@127.0.0.1:{port}/postgres"
    handle = PgHandle(dsn=dsn, kind="local")
    handle._cleanup.append(
        lambda: subprocess.run(
            [str(pg_ctl), "-D", str(data), "-m", "fast", "stop"],
            capture_output=True,
            timeout=60,
            env=env,
        )
    )
    return handle


def _start_pgserver(workdir: Path) -> PgHandle | None:
    try:
        import pgserver
    except ImportError:
        return None
    try:
        server = pgserver.get_server(str(workdir / "pgserver"))
    except Exception:  # noqa: BLE001
        return None
    handle = PgHandle(dsn=server.get_uri(), kind="pgserver")
    handle._cleanup.append(server.cleanup)
    return handle


def _start_docker() -> PgHandle | None:
    if not shutil.which("docker"):
        return None
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        return None
    try:
        container = PostgresContainer("pgvector/pgvector:pg16")
        container.start()
    except Exception:  # noqa: BLE001
        return None
    dsn = container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
    handle = PgHandle(dsn=dsn, kind="docker")
    handle._cleanup.append(container.stop)
    return handle


def acquire() -> PgHandle | None:
    """Start (or find) a Postgres. Returns None if none is possible on this machine."""
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return None
    env = os.environ.get("TOMBSTONE_PG_DSN")
    handle: PgHandle | None = None
    if env:
        handle = PgHandle(dsn=env, kind="env")
    else:
        workdir = Path(tempfile.mkdtemp(prefix="tombstone-pg-"))
        for d in _CANDIDATE_BIN_DIRS:
            if d and Path(d).is_dir():
                handle = _start_local(Path(d), workdir)
                if handle:
                    # A server without pgvector is useless for the vector tests; keep looking.
                    try:
                        hv, _hs, _v = _probe(handle.dsn)
                    except Exception:  # noqa: BLE001
                        hv = False
                    if hv:
                        break
                    handle.stop()
                    handle = None
                    shutil.rmtree(workdir / "pgdata", ignore_errors=True)
        if handle is None:
            pgctl = shutil.which("pg_ctl")
            if pgctl:
                handle = _start_local(Path(pgctl).parent, workdir)
        if handle is None:
            handle = _start_pgserver(workdir)
        if handle is None:
            handle = _start_docker()
        if handle is not None:
            handle._cleanup.append(lambda: shutil.rmtree(workdir, ignore_errors=True))
    if handle is None:
        return None
    # A handle whose server never answers is worse than no handle: the fixture skips on None, but
    # on a dead handle every pgvector test fails with a socket error instead. That is what a
    # contributor without Postgres saw, and what made the mandatory CI jobs red.
    for _ in range(20):
        try:
            handle.has_vector, handle.has_pgstattuple, handle.version = _probe(handle.dsn)
            return handle
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    handle.stop()
    return None
