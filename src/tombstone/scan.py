"""``tombstone scan``: what is in a store that Tombstone has never touched.

Every other command in this tool needs lineage, and lineage only runs forward — it describes
what arrived after the capture hook went on. That leaves the person with a deletion request
today, and three years of vectors already in their database, with nothing to run.

This is what they can run. It opens a store, samples it, and answers one question: **if a
deletion request arrived right now, what could and could not be established?** No config, no
integration.

On writes, precisely, because the difference matters to anyone pointing this at production: this
command issues no insert, update or delete, and the Chroma adapter skips even the segment scrub
it normally performs on open. It cannot promise your files are untouched, because *opening* a
store is enough to make some engines write — Chroma rewrites its index header and its SQLite file
on every open, with or without this tool. So the scan takes a census of the files before and
after and reports what changed, rather than claiming something it does not control.

What it measures, and nothing beyond it:

* how many entries the store holds;
* how many of a sample carry a Tombstone stamp — the marker capture leaves on every record it
  writes, so this is coverage without needing the lineage database at all;
* whether byte-level verification is even possible against this store from this process;
* how many sampled entries share an embedding fingerprint with another sampled entry, because
  those are the ones no byte scan could attribute to one subject *even with* full lineage.

What it cannot tell you, and says so: which entries belong to which person. That is the thing
lineage is for, and no amount of scanning recovers it after the fact.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tombstone.lineage.stamp import K_EMBED, K_SUBJECT
from tombstone.model.status import VerifyLevel
from tombstone.util import fingerprint_f32

DEFAULT_SAMPLE = 200


def _census(root: Path) -> dict[str, tuple[int, int]]:
    """Size and modification time of every file under ``root``. Cheap, and enough to notice a
    rewrite: a write moves the mtime even when the size is unchanged, which is exactly what
    Chroma does to its index header."""
    out: dict[str, tuple[int, int]] = {}
    if not root.exists():
        return out
    for f in root.rglob("*"):
        try:
            if f.is_file():
                st = f.stat()
                out[str(f)] = (st.st_size, st.st_mtime_ns)
        except OSError:
            continue
    return out


@dataclass
class StoreScan:
    """One store, looked at and not touched."""

    name: str
    kind: str
    location: str
    entries: int = 0
    sampled: int = 0
    stamped: int = 0
    physical: bool = False
    physical_reason: str = ""
    duplicate_sampled: int = 0
    dims: int = 0
    version: str = ""
    error: str = ""

    @property
    def coverage(self) -> float | None:
        """Fraction of the sample carrying a Tombstone stamp. None when nothing was sampled."""
        return self.stamped / self.sampled if self.sampled else None

    @property
    def duplicate_rate(self) -> float | None:
        return self.duplicate_sampled / self.sampled if self.sampled else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "location": self.location,
            "entries": self.entries,
            "sampled": self.sampled,
            "stamped": self.stamped,
            "coverage": self.coverage,
            "physical": self.physical,
            "physical_reason": self.physical_reason,
            "duplicate_sampled": self.duplicate_sampled,
            "duplicate_rate": self.duplicate_rate,
            "dims": self.dims,
            "version": self.version,
            "error": self.error,
        }


@dataclass
class ScanReport:
    root: str
    stores: list[StoreScan] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    engine_wrote: list[str] = field(default_factory=list)

    @property
    def total_entries(self) -> int:
        return sum(s.entries for s in self.stores)

    @property
    def total_stamped(self) -> int:
        return sum(s.stamped for s in self.stores)

    @property
    def any_traceable(self) -> bool:
        return self.total_stamped > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "stores": [s.to_dict() for s in self.stores],
            "notes": list(self.notes),
            "total_entries": self.total_entries,
            "any_traceable": self.any_traceable,
            "engine_wrote": list(self.engine_wrote),
        }


# --- looking at one store ------------------------------------------------------------------------


def scan_store(store: Any, name: str, kind: str, location: str, sample: int) -> StoreScan:
    """Sample an open store. Read-only: nothing here writes, deletes or updates."""
    out = StoreScan(name=name, kind=kind, location=location)
    try:
        out.version = str(store.version())
    except Exception as e:  # a version string is never worth failing a scan for
        out.version = f"unknown ({type(e).__name__})"
    try:
        out.entries = int(store.count())
    except Exception as e:
        out.error = f"could not count entries: {type(e).__name__}: {e}"
        return out

    caps: frozenset[VerifyLevel] = getattr(store, "capabilities", frozenset())
    out.physical = VerifyLevel.PHYSICAL in caps
    if not out.physical:
        reason = getattr(store, "physical_unsupported_reason", lambda: "")()
        out.physical_reason = reason or "this store cannot be checked at the byte level"

    if out.entries == 0:
        return out

    try:
        keys = store.sample_keys(min(sample, out.entries))
    except Exception as e:
        out.error = f"could not sample: {type(e).__name__}: {e}"
        return out

    # Suppressed records are included deliberately: a store half-way through an erasure is
    # exactly the state someone might be scanning, and hiding those would flatter the coverage.
    try:
        hits = store.get(keys, include_suppressed=True)
    except TypeError:
        hits = store.get(keys)
    except Exception as e:
        out.error = f"could not read sampled entries: {type(e).__name__}: {e}"
        return out

    out.sampled = len(hits)
    fingerprints: dict[str, int] = {}
    for key, hit in hits.items():
        md = dict(getattr(hit, "metadata", {}) or {})
        if md.get(K_EMBED) or md.get(K_SUBJECT):
            out.stamped += 1
        vec = None
        with contextlib.suppress(Exception):  # a vector we cannot read is a gap, not a failure
            vec = store._vector_of(key) if hasattr(store, "_vector_of") else None
        if vec:
            out.dims = out.dims or len(vec)
            fp = fingerprint_f32(vec)
            fingerprints[fp] = fingerprints.get(fp, 0) + 1
    out.duplicate_sampled = sum(n for n in fingerprints.values() if n > 1)
    return out


# --- finding stores to look at -------------------------------------------------------------------


def _open_configured(config: str | Path | None) -> tuple[list[tuple[Any, str, str, str]], str]:
    """Stores named in a tombstone.yaml. Returns (store, name, kind, location) and the root."""
    from tombstone.registry import Runtime

    rt = Runtime.shared(config)
    found: list[tuple[Any, str, str, str]] = []
    for name, store in rt.all_stores().items():
        loc = str(getattr(store, "path", "") or getattr(store, "dsn", "") or "")
        found.append((store, name, getattr(store, "kind", "?"), _redact(loc)))
    return found, str(rt.inst.root)


def _open_detected(root: Path) -> list[tuple[Any, str, str, str]]:
    """Stores found on disk, with no config and no lineage — the day-zero case."""
    from tombstone.commands.init import detect_project

    det = detect_project(root)
    found: list[tuple[Any, str, str, str]] = []

    for rel in det.chroma_dirs:
        path = root / rel
        try:
            import chromadb
            from chromadb.config import Settings

            from tombstone.stores.chroma import ChromaStore

            client = chromadb.PersistentClient(
                path=str(path), settings=Settings(anonymized_telemetry=False)
            )
            for coll in client.list_collections():
                cname = coll.name if hasattr(coll, "name") else str(coll)
                store = ChromaStore(f"chroma:{cname}", path, collection=cname, read_only=True)
                found.append((store, f"chroma:{cname}", "chroma", str(path)))
        except Exception:  # an unreadable store is reported, never fatal
            found.append((None, f"chroma:{rel}", "chroma", str(path)))

    for rel in det.faiss_files:
        path = root / rel
        try:
            from tombstone.stores.faiss import FaissStore

            found.append(
                (FaissStore(f"faiss:{path.stem}", path), f"faiss:{path.stem}", "faiss", str(path))
            )
        except Exception:
            found.append((None, f"faiss:{path.stem}", "faiss", str(path)))

    for rel in det.qdrant_dirs:
        path = root / rel
        try:
            from tombstone.stores.qdrant import QdrantStore

            found.append(
                (
                    QdrantStore(f"qdrant:{path.name}", path=path),
                    f"qdrant:{path.name}",
                    "qdrant",
                    str(path),
                )
            )
        except Exception:
            found.append((None, f"qdrant:{path.name}", "qdrant", str(path)))

    return found


def _redact(location: str) -> str:
    """A DSN can carry a password, and a scan report is something people paste into tickets."""
    if "://" not in location:
        return location
    scheme, _, rest = location.partition("://")
    if "@" in rest:
        creds, _, host = rest.partition("@")
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return location


def scan(
    root: Path | None = None,
    config: str | Path | None = None,
    dsn: str | None = None,
    table: str = "documents",
    sample: int = DEFAULT_SAMPLE,
) -> ScanReport:
    """Look at every store we can find. Never writes to any of them."""
    root = (root or Path.cwd()).resolve()
    report = ScanReport(root=str(root))
    opened: list[tuple[Any, str, str, str]] = []
    # before anything is opened: opening is itself enough to make some engines write
    before = _census(root)

    cfg = Path(config) if config else root / "tombstone.yaml"
    if Path(cfg).is_file():
        try:
            opened, _r = _open_configured(cfg)
            report.notes.append(f"read the store list from {cfg}")
        except Exception as e:
            report.notes.append(
                f"{cfg} could not be loaded ({type(e).__name__}); scanned the disk instead"
            )
    if not opened:
        opened = _open_detected(root)
        if opened:
            report.notes.append("no usable config: these stores were found on disk")

    if dsn:
        try:
            from tombstone.stores.pgvector import PgVectorStore

            opened.append(
                (
                    PgVectorStore("pgvector", dsn, table),
                    f"pgvector:{table}",
                    "pgvector",
                    _redact(dsn),
                )
            )
        except Exception as e:
            report.notes.append(f"could not open the given DSN: {type(e).__name__}: {e}")

    for store, name, kind, loc in opened:
        if store is None:
            report.stores.append(
                StoreScan(name=name, kind=kind, location=loc, error="could not be opened")
            )
            continue
        report.stores.append(scan_store(store, name, kind, loc, sample))

    if not report.stores:
        report.notes.append(
            "nothing found. Point this at a directory holding a Chroma, FAISS or Qdrant store, "
            "or pass --dsn for Postgres."
        )
    after = _census(root)
    report.engine_wrote = sorted(
        path for path, stat in after.items() if path in before and before[path] != stat
    ) + sorted(p for p in after if p not in before)
    return report


# --- saying what it means ------------------------------------------------------------------------


def render(report: ScanReport, colour: bool = False) -> str:
    def dim(s: str) -> str:
        return f"\x1b[2m{s}\x1b[0m" if colour else s

    def red(s: str) -> str:
        return f"\x1b[31m{s}\x1b[0m" if colour else s

    lines = [f"scanned {report.root}", ""]
    if not report.stores:
        lines += [dim("  " + n) for n in report.notes]
        return "\n".join(lines)

    for s in report.stores:
        lines.append(f"  {s.name}   {dim(s.location)}")
        if s.error:
            lines.append(f"    {red(s.error)}")
            lines.append("")
            continue
        lines.append(f"    entries            {s.entries:,}")
        if s.sampled:
            cov = s.coverage or 0.0
            cov_s = f"{s.stamped}/{s.sampled} sampled ({cov * 100:.0f}%)"
            lines.append(
                f"    tombstone stamps   {cov_s if cov else red(cov_s)}"
                + ("" if cov else "  — nothing here arrived through Tombstone")
            )
            if s.duplicate_sampled:
                rate = (s.duplicate_rate or 0) * 100
                lines.append(
                    f"    shared fingerprints {s.duplicate_sampled}/{s.sampled} sampled "
                    f"({rate:.0f}%) — byte-identical to another entry"
                )
        lines.append(
            "    byte-level proof   "
            + ("available" if s.physical else red("not available") + f" — {s.physical_reason}")
        )
        lines.append("")

    lines.append("If a deletion request arrived today:")
    if report.any_traceable:
        lines.append("  ✓ some of this is traceable — run `tombstone trace --subject <id>`")
        lines.append("  ! the unstamped remainder has no trail and cannot be claimed as erased")
    else:
        lines.append("  ✗ nothing here can be traced to a person by this tool.")
        lines.append("    Tombstone follows what it watched arrive; it cannot reconstruct")
        lines.append("    afterwards which entries came from whom.")
        lines.append("  ✓ if your application records which ids belong to whom, you can still")
        lines.append("    delete by id — and the byte-level check below tells you whether you")
        lines.append("    could then prove the bytes are gone.")
    dupes = [s for s in report.stores if s.duplicate_sampled]
    if dupes:
        lines.append(
            "  ! some entries are byte-identical to others, so no scan could attribute them"
        )
        lines.append("    to one person even with full lineage. That is a fact about the data.")
    noproof = [s for s in report.stores if not s.physical and not s.error]
    if noproof:
        lines.append(
            "  ! "
            + ", ".join(s.name for s in noproof)
            + " cannot be checked at the byte level from here"
        )
    lines.append("")
    lines.append("Next: `tombstone init` writes a config, then wrap the store at ingest so")
    lines.append("      everything from now on is traceable.")
    lines.append("")
    lines.append("This scan issued no insert, update or delete. Opening a store is not free of")
    if report.engine_wrote:
        n = len(report.engine_wrote)
        lines.append(
            f"writes, though: the database engine rewrote {n} of its own file(s) while we read"
        )
        lines.append("it, which it does on any open, by any client:")
        for f in report.engine_wrote[:5]:
            lines.append(f"  {dim(f)}")
        if n > 5:
            lines.append(dim(f"  … and {n - 5} more"))
    else:
        lines.append("writes in general, but here nothing on disk changed while we looked.")
    for note in report.notes:
        lines.append(dim(f"  ({note})"))
    return "\n".join(lines)


__all__ = ["ScanReport", "StoreScan", "render", "scan", "scan_store"]
