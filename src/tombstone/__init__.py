"""Tombstone: lineage-driven erasure of a data subject's derived artifacts.

The core package imports nothing heavier than ``pydantic``, ``cryptography`` and the
standard library. Vector backends, LangChain, training and MCP are optional extras and are
imported lazily by the modules that need them.

The public API is two functions::

    from tombstone import trace, forget

    held = trace("S-0417")             # what do we hold on this person?
    result = forget("S-0417", reason="dsr-2026-0912")   # destructive; returns a receipt

Both are resolved lazily (PEP 562) so ``import tombstone`` still costs nothing: the runtime, the
config and the store adapters load on first use, not on import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tombstone.api import Erasure, Held, forget, trace

__all__ = ["Erasure", "Held", "__version__", "forget", "trace"]

_LAZY = {"forget", "trace", "Held", "Erasure"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from tombstone import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
