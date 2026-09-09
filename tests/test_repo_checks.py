"""9.1: banned words never appear in README, docs, src or outputs; README numbers inside measured
blocks trace to committed result files."""

from __future__ import annotations

from pathlib import Path

from tombstone._checks import check_banned_words, check_readme_numbers

ROOT = Path(__file__).resolve().parents[1]


def test_no_banned_words_in_repo() -> None:
    hits = check_banned_words(
        [
            str(ROOT / p)
            for p in ("README.md", "src", "docs", "RESULTS.md", "CHANGELOG.md")
            if (ROOT / p).exists()
        ]
    )
    assert not hits, "\n".join(hits)


def test_readme_numbers_trace_to_results() -> None:
    problems = check_readme_numbers(ROOT / "README.md", ROOT / "bench" / "results")
    assert not problems, "\n".join(problems)


def test_banned_word_detector_has_power(tmp_path: Path) -> None:
    f = tmp_path / "x.md"
    f.write_text("This erasure is Certified and GDPR compliant.\n")
    hits = check_banned_words([str(f)])
    assert len(hits) == 2


def test_readme_number_detector_has_power(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r.json").write_text('{"rate": 0.427, "n": 200}')
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- measured:start -->\n42.7% of 200 subjects, and 99 others\n<!-- measured:end -->\nPython 3.12 is fine outside.\n"
    )
    problems = check_readme_numbers(readme, tmp_path / "results")
    assert len(problems) == 1 and "'99'" in problems[0]
