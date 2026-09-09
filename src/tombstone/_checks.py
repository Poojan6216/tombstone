"""Repository checks run by CI and pre-commit: banned words (Hard Rule 2) and README traceability.

Usage: ``python -m tombstone._checks banned-words [paths...]``
       ``python -m tombstone._checks readme-numbers README.md bench/results``
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BANNED = (
    "certified",
    "guaranteed",
    "complete erasure",
    "gdpr compliant",
    "gdpr-compliant",
    "enterprise-grade",
)
# Words are checked case-insensitively. "certificate" is also excluded from outputs by policy,
# but "certificate" legitimately appears in dependency names (TLS), so only the exact banned
# terms above are enforced mechanically.

_DEFAULT_PATHS = ("README.md", "src", "docs", "RESULTS.md", "CHANGELOG.md")
_SKIP_DIRS = {".venv", "node_modules", "__pycache__", ".git", ".tombstone"}
_ALLOW_MARK = "banned-words: allow"  # a line carrying this marker is exempt (prior-art quotes)


def iter_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for f in path.rglob("*"):
                if (
                    f.is_file()
                    and not any(part in _SKIP_DIRS for part in f.parts)
                    and f.suffix
                    in {".py", ".md", ".txt", ".yaml", ".yml", ".sql", ".json", ".html"}
                ):
                    out.append(f)
        elif path.is_file():
            out.append(path)
    return out


def check_banned_words(paths: list[str]) -> list[str]:
    hits: list[str] = []
    for f in iter_files(paths):
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if _ALLOW_MARK in line or f.name in {"_checks.py", "BUILD_SPEC.md"}:
                continue
            low = line.lower()
            for word in BANNED:
                if word in low:
                    hits.append(f"{f}:{i}: banned word {word!r}")
    return hits


_NUM_RE = re.compile(r"(?<![\w./-])(\d+(?:\.\d+)?)(?:%)?(?![\w./-])")


def _collect_result_numbers(results_dir: Path) -> set[str]:
    """Every numeric literal (as rendered by str/round) appearing in committed result JSON."""
    found: set[str] = set()

    def walk(x: object) -> None:
        if isinstance(x, bool):
            return
        if isinstance(x, int):
            found.add(str(x))
        elif isinstance(x, float):
            found.add(repr(x))
            for nd in (0, 1, 2, 3, 4):
                found.add(f"{x:.{nd}f}")
                found.add(f"{x * 100:.{nd}f}")
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and x.replace(".", "", 1).isdigit():
            found.add(x)

    for f in results_dir.rglob("*.json"):
        try:
            walk(json.loads(f.read_text(encoding="utf-8")))
        except (ValueError, UnicodeDecodeError):
            continue
    return found


def check_readme_numbers(readme: Path, results_dir: Path) -> list[str]:
    """Numbers inside result-bearing lines must trace to a committed JSON file.

    Only lines marked with ``<!-- measured -->`` (or inside a ``<!-- measured:start -->`` block)
    are checked — prose like "Python 3.12" or "Article 17" is not a measurement.
    """
    allowed = _collect_result_numbers(results_dir)
    problems: list[str] = []
    in_block = False
    for i, line in enumerate(readme.read_text(encoding="utf-8").splitlines(), 1):
        if "<!-- measured:start -->" in line:
            in_block = True
            continue
        if "<!-- measured:end -->" in line:
            in_block = False
            continue
        if not (in_block or "<!-- measured -->" in line):
            continue
        for m in _NUM_RE.finditer(line):
            token = m.group(1)
            if token not in allowed:
                problems.append(f"{readme}:{i}: number {token!r} not found in {results_dir}")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        sys.stderr.write(__doc__ or "")
        return 2
    cmd, rest = args[0], args[1:]
    if cmd == "banned-words":
        hits = check_banned_words(rest or list(_DEFAULT_PATHS))
        for h in hits:
            sys.stderr.write(h + "\n")
        return 1 if hits else 0
    if cmd == "readme-numbers":
        readme = Path(rest[0]) if rest else Path("README.md")
        results = Path(rest[1]) if len(rest) > 1 else Path("bench/results")
        problems = check_readme_numbers(readme, results)
        for p in problems:
            sys.stderr.write(p + "\n")
        return 1 if problems else 0
    sys.stderr.write(f"unknown check {cmd!r}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
