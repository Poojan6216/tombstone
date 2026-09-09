"""Every failure this tool can raise on purpose. Silence is the failure mode we exist to remove."""

from __future__ import annotations


class TombstoneError(Exception):
    """Base class. Exit code 1 from the CLI."""

    exit_code = 1


class ConfigError(TombstoneError):
    """A config problem, naming the field and (where possible) the line."""


class LineageGapError(TombstoneError):
    """Hard Rule 4: an erasure/trace hit a subject or store with missing lineage."""


class ScopeViolation(TombstoneError):
    """Hard Rule 8: an edge crosses tenant scopes."""


class PinMismatch(TombstoneError):
    """Hard Rule (2.4): a store/model/manifest changed silently since it was pinned."""


class NotSupported(TombstoneError):
    """A store cannot perform a probe at the requested level. Message is actionable."""


class SagaError(TombstoneError):
    """The erasure saga halted (Hard Rule 6/10)."""


class ChainBroken(TombstoneError):
    """A hash chain (ledger or journal) fails verification. Message names the record index."""


class SignatureInvalid(TombstoneError):
    """A receipt signature does not verify."""


class ReplayMismatch(TombstoneError):
    """Hard Rule 9: replay re-derived a different status table."""

    exit_code = 3


class ConfirmationRequired(TombstoneError):
    """A destructive operation was requested without explicit confirmation."""

    exit_code = 2


class LockTimeout(TombstoneError):
    """Could not acquire the journal/ledger lock within the timeout."""
