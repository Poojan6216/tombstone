"""A local web page for the people who actually handle deletion requests.

Support and legal receive the DSR, and they do not use a terminal. ``tombstone ui`` serves one
page on loopback: search a person, see what is held, erase it, read the receipt.

This server can delete data, so it is deliberately hard to reach:

* it binds 127.0.0.1 only — there is no flag to serve it on a network interface;
* every ``/api/`` request must carry a random per-run token in the ``X-Tombstone-Token`` header.
  The token is in the URL printed at startup, and the page moves it out of the address bar into
  memory on load. A header cannot be set by a cross-origin form post, so a page the operator
  happens to be browsing cannot drive this one, and there is no cookie for a browser to attach
  automatically;
* ``Origin``, when the browser sends one, must match the server's own;
* nothing is cached, and the page is served with a restrictive CSP.

Erasure still refuses lineage gaps, still asks for a reason, and still writes the same journal
and receipt as the CLI. The button is a nicer way to answer the same question, not a shortcut
past it.
"""

from __future__ import annotations

from tombstone.ui.server import serve

__all__ = ["serve"]
