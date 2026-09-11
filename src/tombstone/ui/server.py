"""The loopback HTTP server behind ``tombstone ui``. Standard library only."""

from __future__ import annotations

import hmac
import json
import logging
import secrets
import sys
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from tombstone.api import erase_traced, look
from tombstone.errors import TombstoneError
from tombstone.logging import get_logger
from tombstone.registry import Runtime
from tombstone.ui.page import PAGE

_log = get_logger("ui")

HOST = "127.0.0.1"
TOKEN_HEADER = "X-Tombstone-Token"  # noqa: S105 - a header name, not a secret
MAX_BODY = 64 * 1024

# No inline scripts or styles from anywhere but this document, nothing loaded from the network,
# no framing. The page is self-contained, so this costs nothing.
CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


class _Api:
    """The four things the page can ask for. One runtime, one lock: a local tool serving one
    operator does not need concurrency, and SQLite is happier without it."""

    def __init__(self, config: str | Path | None) -> None:
        self.config = config
        self._lock = threading.Lock()

    def _rt(self) -> Runtime:
        return Runtime.shared(self.config)

    def status(self, _body: dict[str, Any]) -> dict[str, Any]:
        from tombstone.commands.trace import status_payload

        with self._lock:
            return status_payload(self._rt())

    def trace(self, body: dict[str, Any]) -> dict[str, Any]:
        subject = str(body.get("subject", "")).strip()
        if not subject:
            raise ValueError("a subject id is required")
        with self._lock:
            held, _t = look(self._rt(), subject, scan_stores=bool(body.get("scan_stores", True)))
        return held.to_dict()

    def forget(self, body: dict[str, Any]) -> dict[str, Any]:
        subject = str(body.get("subject", "")).strip()
        reason = str(body.get("reason", "")).strip()
        if not subject:
            raise ValueError("a subject id is required")
        if not reason:
            raise ValueError("a reason is required; it goes on the receipt")
        with self._lock:
            rt = self._rt()
            held, t = look(rt, subject, scan_stores=True)
            # The page shows the operator what it found and asks them to type a reason. What it
            # must not do is let that click mean more than the CLI's would: the same gap refusal
            # applies, and accepting gaps is a separate, explicit choice.
            if held.gaps and not bool(body.get("accept_gaps", False)):
                raise TombstoneError(
                    f"{len(held.gaps)} lineage gap(s): "
                    + "; ".join(held.gaps)
                    + ". Nothing was changed. Tick 'accept lineage gaps' to proceed and record "
                    "UNVERIFIED(lineage-gap) on the receipt."
                )
            if body.get("expect_trace_id") and body["expect_trace_id"] != t.trace_id:
                raise TombstoneError(
                    "the lineage graph changed since this page last looked. Nothing was changed; "
                    "search again to see what is there now."
                )
            result = erase_traced(rt, t, reason, accept_gaps=bool(body.get("accept_gaps", False)))
        out = result.to_dict()
        out["report"] = result.report
        return out

    def receipts(self, _body: dict[str, Any]) -> dict[str, Any]:
        from tombstone.receipt.ledger import Ledger

        with self._lock:
            rt = self._rt()
            ledger = Ledger(rt.inst.ledger_path)
            rows = [
                {
                    "receipt_id": r.receipt_id,
                    "subject": r.subject.short,
                    "reason": r.reason,
                    "created_ms": r.created_ms,
                    "counts": {k.value: int(v) for k, v in r.counts.items()},
                }
                for r in ledger.receipts()
            ]
            chain_ok = ledger.verify() >= 0
        rows.reverse()
        return {"receipts": rows, "chain_ok": chain_ok}


def _handler(api: _Api, token: str) -> type[BaseHTTPRequestHandler]:
    routes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
        "/api/status": api.status,
        "/api/trace": api.trace,
        "/api/forget": api.forget,
        "/api/receipts": api.receipts,
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "tombstone"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # keep stdout clean
            _log.debug("ui %s", fmt % args)

        # --- helpers ---------------------------------------------------------------------

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json; charset=utf-8")

        def _authorised(self) -> bool:
            sent = self.headers.get(TOKEN_HEADER, "")
            if not hmac.compare_digest(sent, token):
                return False
            origin = self.headers.get("Origin")
            address = self.server.server_address
            port = address[1] if isinstance(address, tuple) else 0
            return not origin or origin in {
                f"http://{HOST}:{port}",
                f"http://localhost:{port}",
            }

        # --- routes ----------------------------------------------------------------------

        def do_GET(self) -> None:  # BaseHTTPRequestHandler's spelling, not ours
            path = self.path.split("?", 1)[0]
            if path in {"/", "/index.html"}:
                self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            route = routes.get(path)
            if route is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if not self._authorised():
                # Deliberately uninformative: this is the wall between a web page the operator
                # is browsing and their deletion tool.
                self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body too large"})
                return
            try:
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as e:
                self._json(HTTPStatus.BAD_REQUEST, {"error": f"bad request: {e}"})
                return
            try:
                self._json(HTTPStatus.OK, route(body))
            except TombstoneError as e:
                self._json(HTTPStatus.CONFLICT, {"error": str(e)})
            except ValueError as e:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            except Exception as e:  # never leak a traceback to the browser
                _log.exception("ui request failed", extra={"tombstone_extra": {"path": path}})
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"{type(e).__name__}: {e}"},
                )

    return Handler


def build_server(
    config: str | Path | None, port: int = 0, token: str | None = None
) -> tuple[HTTPServer, str]:
    """An HTTP server bound to loopback, and the token its API requires. Port 0 picks a free one."""
    tok = token or secrets.token_urlsafe(32)
    httpd = HTTPServer((HOST, port), _handler(_Api(config), tok))
    return httpd, tok


def serve(config: str | Path | None, port: int = 7878, open_browser: bool = True) -> int:
    """Run until interrupted. Returns a process exit code."""
    logging.getLogger("tombstone.ui").setLevel(logging.WARNING)
    httpd, token = build_server(config, port)
    bound = httpd.server_address[1]
    url = f"http://{HOST}:{bound}/?t={token}"
    sys.stdout.write(
        f"tombstone ui  →  {url}\n"
        "This page can erase data. The link carries a token generated for this run only;\n"
        "anyone who has it can use the page. Stop the server with Ctrl-C.\n"
    )
    sys.stdout.flush()
    if open_browser:
        import webbrowser

        with_thread = threading.Timer(0.2, lambda: webbrowser.open(url))
        with_thread.daemon = True
        with_thread.start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stdout.write("\nstopped.\n")
    finally:
        httpd.server_close()
    return 0
