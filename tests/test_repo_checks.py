"""9.1: banned words never appear in README, docs, src or outputs; README numbers inside measured
blocks trace to committed result files."""

from __future__ import annotations

import ast
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


# --- Hard Rule 1: no LLM in the decision or verification path -------------------------------------

DECISION_PACKAGES = ("lineage", "erase", "verify")
LLM_MODULES = frozenset(
    {
        "openai",
        "anthropic",
        "transformers",
        "torch",
        "peft",
        "trl",
        "sentence_transformers",
        "langchain_openai",
        "langchain_anthropic",
        "litellm",
        "ollama",
        "cohere",
        "vllm",
        "llama_cpp",
        "huggingface_hub",
        "vec2text",
    }
)
INVOCATION_ATTRS = frozenset({"generate", "chat", "invoke", "complete", "completions", "predict"})


def _module_imports(path: Path) -> set[str]:
    """Top-level module names imported by ``path`` (absolute imports only)."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module.split(".")[0])
    return out


def _invocation_calls(path: Path) -> set[str]:
    """Calls shaped like a model invocation (``x.generate(...)``, ``x.chat(...)``)."""
    return {
        node.func.attr
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in INVOCATION_ATTRS
    }


def _decision_path_files() -> list[Path]:
    return sorted(
        f for pkg in DECISION_PACKAGES for f in (ROOT / "src" / "tombstone" / pkg).glob("*.py")
    )


def test_no_llm_in_the_decision_or_verification_path() -> None:
    """Hard Rule 1: which artifacts to delete is a graph query and whether one is gone is a
    deterministic probe, so `lineage/`, `erase/` and `verify/` import no model library and call
    nothing shaped like a model invocation.

    ``verify/audit.py`` takes ``probe_prompts``, but it neither builds nor sends one: the pairs are
    the app's own canaries, forwarded to the adapter store (``stores/``), which is the artifact
    under test rather than a judge of whether the data is gone.
    """
    files = _decision_path_files()
    assert files, "no modules found on the decision path — the glob is wrong"
    offenders = []
    for f in files:
        bad_imports = _module_imports(f) & LLM_MODULES
        bad_calls = _invocation_calls(f)
        if bad_imports or bad_calls:
            rel = f.relative_to(ROOT)
            offenders.append(f"{rel}: imports={sorted(bad_imports)} calls={sorted(bad_calls)}")
    assert not offenders, "LLM reached the decision/verification path: " + "; ".join(offenders)


def test_llm_detector_has_power(tmp_path: Path) -> None:
    """The detector above must actually catch an LLM sneaking in."""
    bad = tmp_path / "sneaky.py"
    bad.write_text(
        "import transformers\n\n\ndef judge(model, a):\n    return model.generate('is it gone?')\n",
        encoding="utf-8",
    )
    assert _module_imports(bad) & LLM_MODULES == {"transformers"}
    assert _invocation_calls(bad) == {"generate"}
    clean = tmp_path / "fine.py"
    clean.write_text("import json\n\n\ndef f(x):\n    return json.dumps(x)\n", encoding="utf-8")
    assert not _module_imports(clean) & LLM_MODULES
    assert not _invocation_calls(clean)
