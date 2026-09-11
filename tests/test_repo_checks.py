"""9.1: banned words never appear in README, docs, src or outputs; README numbers inside measured
blocks trace to committed result files."""

from __future__ import annotations

import ast
from pathlib import Path

from tombstone._checks import (
    check_banned_words,
    check_docs_have_measured_numbers,
    check_readme_numbers,
)

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


TELEMETRY_MODULES = frozenset(
    {
        "posthog",
        "sentry_sdk",
        "mixpanel",
        "amplitude",
        "analytics",
        "statsd",
        "datadog",
        "ddtrace",
        "opentelemetry",
        "segment",
        "bugsnag",
        "rollbar",
    }
)


def test_no_telemetry_and_no_hosted_components() -> None:
    """Nothing in the package reports usage anywhere, and there is no endpoint to report to.

    Tombstone runs entirely on the operator's machine: the only hosts it ever talks to are the
    operator's own stores, named in their own config. So no analytics SDK may be imported, and no
    URL may be hard-coded outside a docstring citation.

    One exception, and it is the opposite of phoning home: ``tombstone ui`` serves a page to the
    operator's own browser and prints its address. Loopback hosts are allowed anywhere; the bare
    ``http://`` fragment an f-string leaves behind is allowed only in that server, whose bind
    address is pinned to loopback at the bottom of this test. Anything naming a remote host still
    fails, here and in any file added later.
    """
    ui_server = Path("src/tombstone/ui/server.py")
    offenders: list[str] = []
    urls: list[str] = []
    for f in sorted((ROOT / "src" / "tombstone").rglob("*.py")):
        bad = _module_imports(f) & TELEMETRY_MODULES
        if bad:
            offenders.append(f"{f.relative_to(ROOT)}: {sorted(bad)}")
        tree = ast.parse(f.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        rel = f.relative_to(ROOT)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and ("http://" in node.value or "https://" in node.value)
                and not _is_loopback(node.value, rel == ui_server)
            ):
                urls.append(f"{rel}:{node.lineno}: {node.value[:60]}")
    assert not offenders, "telemetry SDK imported: " + "; ".join(offenders)
    assert not urls, "hard-coded URL outside a docstring: " + "; ".join(urls)
    # the exception above is only sound while the UI cannot be served off the loopback interface
    from tombstone.ui.server import HOST, build_server

    assert HOST == "127.0.0.1", HOST
    httpd, _token = build_server(None, port=0)
    try:
        assert httpd.server_address[0] == "127.0.0.1", httpd.server_address
    finally:
        httpd.server_close()
    ui_src = (ROOT / ui_server).read_text(encoding="utf-8")
    assert "0.0.0.0" not in ui_src, "the UI must not offer a non-loopback bind address"


def _is_loopback(value: str, in_ui_server: bool) -> bool:
    """A URL constant that cannot reach anything but this machine."""
    for prefix in ("http://127.0.0.1", "http://localhost", "http://[::1]"):
        if value.startswith(prefix):
            return True
    # f"http://{HOST}:{port}" leaves "http://" as its own constant; only the UI server may do that
    return in_ui_server and value in {"http://", "http://localhost:"}


def test_chroma_phone_home_stays_disabled() -> None:
    """Chroma's client reports anonymous usage by default; the store must keep it switched off."""
    src = (ROOT / "src" / "tombstone" / "stores" / "chroma.py").read_text(encoding="utf-8")
    assert "anonymized_telemetry=False" in src, "Chroma telemetry is no longer disabled"


def test_docs_number_detector_has_power(tmp_path: Path) -> None:
    """9.2's check must catch a doc that cites nothing measured, and must not be satisfied by a
    number that only collides with a result file by accident."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "r.json").write_text('{"rate": 0.725, "subjects": 16}', encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "cites.md").write_text("the attack succeeds 72.5% of the time", encoding="utf-8")
    (docs / "prose.md").write_text("see section 7.5; we ran 16 shards", encoding="utf-8")
    problems = check_docs_have_measured_numbers(docs, results)
    named = {Path(p.split(":")[0]).name for p in problems}
    assert named == {"prose.md"}, problems  # 16 collides by luck; 7.5 is a section number


def test_docs_cite_measured_numbers_that_are_ready() -> None:
    """The real docs, minus the two still waiting on runs in flight.

    ``demo.md`` is regenerated with its own result file once the unlearning adapters exist, and
    ``writeup.md`` (9.5) is written against the final numbers; both are expected to fail until
    then, and this test tightens to the whole directory when they land.
    """
    pending = {"demo.md", "writeup.md"}
    problems = [
        p
        for p in check_docs_have_measured_numbers(ROOT / "docs", ROOT / "bench" / "results")
        if Path(p.split(":")[0]).name not in pending
    ]
    assert not problems, problems
