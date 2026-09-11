"""``tombstone ui`` serves a page that can erase data, so most of what matters here is what it
refuses. The wall is a per-run token in a request header: a page the operator happens to be
browsing can POST to loopback, but it cannot read that token and cannot set a custom header on a
cross-origin request without a preflight this server never approves.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests import _pipeline, _stores
from tombstone.receipt.ledger import Ledger
from tombstone.ui.server import TOKEN_HEADER, build_server


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    monkeypatch.chdir(tmp_path)
    _stores.skip_unless("faiss")
    h = _pipeline.build(tmp_path, ["faiss"], rag_subjects=("S-0001",))
    httpd, token = build_server(h["cfg_path"], port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "base": f"http://127.0.0.1:{httpd.server_address[1]}",
            "token": token,
            "root": tmp_path,
            "corpus": h["corpus"],
        }
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _post(
    base: str, path: str, body: dict[str, Any], token: str | None, origin: str | None = None
) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token is not None:
        req.add_header(TOKEN_HEADER, token)
    if origin is not None:
        req.add_header("Origin", origin)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - our own loopback server
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _receipts(root: Path) -> list[Any]:
    return Ledger(root / ".tombstone" / "ledger.jsonl").receipts()


def test_the_page_is_self_contained(served: dict[str, Any]) -> None:
    with urllib.request.urlopen(served["base"] + "/", timeout=30) as r:  # noqa: S310
        assert r.status == 200
        html = r.read().decode()
        assert "Content-Security-Policy" in dict(r.headers)
        assert r.headers["Cache-Control"] == "no-store"
    # nothing is fetched from the network: no CDN, no font host, no analytics
    for scheme in ("http://", "https://", "//cdn"):
        assert scheme not in html.replace("http://127.0.0.1", ""), scheme


@pytest.mark.parametrize("path", ["/api/trace", "/api/forget", "/api/status", "/api/receipts"])
def test_every_api_route_refuses_without_the_token(served: dict[str, Any], path: str) -> None:
    code, body = _post(served["base"], path, {"subject": "S-0001", "reason": "r"}, token=None)
    assert code == 403 and body["error"] == "forbidden"
    assert not _receipts(served["root"])


def test_a_wrong_token_is_refused(served: dict[str, Any]) -> None:
    code, _ = _post(served["base"], "/api/trace", {"subject": "S-0001"}, token="not-the-token")
    assert code == 403
    # and a prefix of the real one is not "close enough"
    code, _ = _post(served["base"], "/api/trace", {"subject": "S-0001"}, served["token"][:-1])
    assert code == 403


def test_a_foreign_origin_is_refused_even_with_the_token(served: dict[str, Any]) -> None:
    """Belt and braces: if a token ever leaked into a page, the Origin check still stops it."""
    code, _ = _post(
        served["base"],
        "/api/forget",
        {"subject": "S-0001", "reason": "r"},
        served["token"],
        origin="https://evil.example",
    )
    assert code == 403
    assert not _receipts(served["root"])


def test_trace_then_forget_is_the_whole_flow(served: dict[str, Any]) -> None:
    base, token, root = served["base"], served["token"], served["root"]

    code, held = _post(base, "/api/trace", {"subject": "S-0001"}, token)
    assert code == 200, held
    assert held["count"] > 0 and held["trace_id"]
    assert "S-0001" not in json.dumps(held)  # Hard Rule 3: the raw id never comes back

    code, res = _post(
        base,
        "/api/forget",
        {"subject": "S-0001", "reason": "dsr-ui-1", "expect_trace_id": held["trace_id"]},
        token,
    )
    assert code == 200, res
    assert res["receipt_id"] and res["exit_code"] in (0, 2)
    assert res["counts"]["verified"] >= 0 and "VERIFIED" in res["report"]
    receipts = _receipts(root)
    assert [r.receipt_id for r in receipts] == [res["receipt_id"]]
    assert receipts[0].reason == "dsr-ui-1"

    code, past = _post(base, "/api/receipts", {}, token)
    assert code == 200 and [r["receipt_id"] for r in past["receipts"]] == [res["receipt_id"]]
    # the history lists ids and counts, never content
    dumped = json.dumps(past)
    for d in served["corpus"]:
        if d.subject == "S-0001" and getattr(d, "canary", None):
            assert d.canary.token not in dumped


def test_forget_needs_a_reason(served: dict[str, Any]) -> None:
    code, body = _post(served["base"], "/api/forget", {"subject": "S-0001"}, served["token"])
    assert code == 400 and "reason" in body["error"]
    assert not _receipts(served["root"])


def test_a_stale_view_cannot_erase_what_it_did_not_show(served: dict[str, Any]) -> None:
    """The page sends back the trace id it displayed. If the graph moved, the click approved a
    different set of artifacts than the one that would go, so the server refuses."""
    code, body = _post(
        served["base"],
        "/api/forget",
        {"subject": "S-0001", "reason": "r", "expect_trace_id": "01STALETRACEID000000000000"},
        served["token"],
    )
    assert code == 409 and "changed since" in body["error"]
    assert not _receipts(served["root"])


def test_gaps_are_refused_until_explicitly_accepted(served: dict[str, Any]) -> None:
    from tombstone.lineage.capture import EmbedRecord
    from tombstone.registry import Runtime

    base, token, root = served["base"], served["token"], served["root"]
    rt = Runtime.shared(root / "tombstone.yaml")
    store = next(s for s in rt.all_stores().values() if s.kind == "faiss")
    emb = _stores.embedder()
    texts = [f"unstamped {i}" for i in range(20)]
    store._add(  # type: ignore[attr-defined]
        [
            EmbedRecord(f"legacy{i}", v, {}, t, None, None)  # type: ignore[arg-type]
            for i, (v, t) in enumerate(zip(emb.embed(texts), texts, strict=True))
        ]
    )

    code, body = _post(base, "/api/forget", {"subject": "S-0001", "reason": "r"}, token)
    assert code == 409 and "lineage gap" in body["error"]
    assert not _receipts(root)

    code, res = _post(
        base, "/api/forget", {"subject": "S-0001", "reason": "r", "accept_gaps": True}, token
    )
    assert code == 200, res
    assert len(_receipts(root)) == 1


def test_an_unknown_route_is_not_found(served: dict[str, Any]) -> None:
    code, _ = _post(served["base"], "/api/anything", {}, served["token"])
    assert code == 404


def test_the_hide_utility_beats_the_rules_declared_after_it() -> None:
    """The gap checkbox shipped visible when there were no gaps: `.check` is declared below
    `.hide`, so on equal specificity it won and the element never hid. A utility class has to
    beat whatever it sits on regardless of source order."""
    from tombstone.ui.page import PAGE

    assert ".hide{display:none !important}" in PAGE
    # every element the script toggles must start hidden, so nothing flashes before data arrives
    for element_id in ("err", "held", "receipt", "past", "heldHuman", "heldGaps", "gapCheckWrap"):
        marker = f'id="{element_id}"'
        assert marker in PAGE, element_id
        tag_start = PAGE.rindex("<", 0, PAGE.index(marker))
        tag = PAGE[tag_start : PAGE.index(">", tag_start)]
        assert "hide" in tag, f"{element_id} does not start hidden: {tag}"
