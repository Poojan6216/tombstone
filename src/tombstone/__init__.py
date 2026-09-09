"""Tombstone: lineage-driven erasure of a data subject's derived artifacts.

The core package imports nothing heavier than ``pydantic``, ``cryptography`` and the
standard library. Vector backends, LangChain, training and MCP are optional extras and are
imported lazily by the modules that need them.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
