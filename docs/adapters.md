# Adding a store adapter

A store adapter is under 200 lines. It implements four protocol methods plus three small
helpers, detects its own capabilities at connect time, and must pass the shared contract test
in `tests/test_adapter_suite.py`. The in-memory dict store at `src/tombstone/stores/memory.py`
was written from this page alone and is the smallest complete example.

## The protocol (`src/tombstone/stores/base.py`)

```python
class ErasableStore(Protocol):
    name: str                             # the store name from tombstone.yaml
    kind: str                             # "chroma", "pgvector", "memory", ...
    capabilities: frozenset[VerifyLevel]  # what THIS instance can verify

    def suppress(self, refs: Sequence[ArtifactRef]) -> None: ...          # phase 1, idempotent
    def reclaim(self, refs: Sequence[ArtifactRef]) -> ReclaimResult: ...  # phase 2, idempotent
    def probe_logical(self, ref: ArtifactRef, probes: ProbeSet) -> LogicalProbeResult: ...
    def probe_physical(self, ref: ArtifactRef) -> PhysicalProbeResult: ...  # NotSupported if not capable
    def count(self) -> int: ...
    def sample_keys(self, n: int) -> list[str]: ...
    def version(self) -> str: ...
    def close(self) -> None: ...
```

An adapter never decides an outcome. It suppresses, reclaims, and answers probes with facts
(found or not, counts, file names). The status lattice in `src/tombstone/verify/levels.py`
turns those facts into `VERIFIED` / `UNVERIFIED` / `RESIDUAL`.

## What each method must do

**`suppress(refs)`** — make the artifacts unretrievable *immediately* through every query path
the store exposes, without touching the bytes. A metadata flag plus a filter on every query
path, or an exclusion set persisted next to the index. Idempotent: calling it twice is fine.
Caches have no useful "hidden" state, so they delete the entry here.

**`reclaim(refs)`** — remove the bytes. Return `ReclaimResult(noop, method, measurement)`.
`noop` must be decided from residue, not from "nothing left to delete": a reclaim after the app's
own `delete()` still does real work. The second call on the same refs must return `noop=True`.
Record the method string honestly (`"DELETE + REINDEX + VACUUM FULL"`, or
`"DELETE only (VACUUM not permitted)"`).

**`probe_logical(ref, probes)`** — try to get the artifact back: by id, by metadata filter on
`tombstone.embed_id` / `tombstone.artifact_id`, and by top-k with each vector in
`probes.vectors` (plus MMR where the store supports it). Return `found=True` if *any* probe
returns it, with `found_by` naming which. A present artifact must be found (the test checks
probe power).

**`probe_physical(ref)`** — scan the persisted bytes for the artifact's patterns and report
counts per pattern in `measurement` as `matches_<pattern>`:

- `matches_artifact_id` (or `matches_id`, `matches_cache_key`): a pattern that identifies *this*
  artifact's own record. If it is present, the record is present.
- content patterns (`matches_f32le`, `matches_pickle_binfloat`, `matches_hash`, …): bytes that
  byte-identical copies from other subjects would share. The saga compares these against a
  baseline taken before reclaim when lineage shows live duplicates.

Raise `NotSupported` with an actionable message when the instance cannot be checked (no file
access, not the table owner). Never return `found=False` because you could not look.

## Capability detection

Detect at connect, never assume. `{LOGICAL}` is the floor. Add `PHYSICAL` only when the process
can read (and, for reclaim, rewrite) the persisted bytes: a readable persist directory, a
writable index file, `pg_read_binary_file` rights, a snapshot API. Add `SEMANTIC` when the
store supports vector queries and physical access (drift is measured through queries). Add
`MODEL` only for adapters that can run extraction and membership inference. A managed database
reports `{LOGICAL}` and the receipt shows `UNVERIFIED-managed` — that is correct behaviour.

## What each verification level requires from you

| level | the adapter provides | the lattice concludes |
|---|---|---|
| logical | id, filter, top-k and MMR probes all miss | not retrievable through this store's API |
| physical | id pattern absent; content patterns absent or attributable to live duplicates | bytes not present in storage |
| semantic | `query()` for the drift protocol | drift vs same-cluster control, reported only |
| model | extraction and MIA (`probe_model`) | canaries not extractable, MIA at chance |

## Managed services: the adapter that cannot look

`src/tombstone/stores/pinecone.py` is the reference for a backend running on somebody else's
machines. There is no persist directory, no snapshot, no file-read API, so `probe_physical`
raises `NotSupported` and `capabilities` is `{LOGICAL}` — and the receipt says
`UNVERIFIED-managed`. Three rules that adapter follows, and yours should:

1. **Never return `found=False` because you could not look.** "I checked and it is gone" and "I
   cannot check" are different answers. Only one of them is true for a managed service, and
   returning the wrong one turns the tool into the thing it exists to replace.
2. **Say what the operator is actually relying on.** The reason string names the vendor's own
   deletion commitment as the remaining assurance, rather than implying a measurement.
3. **Wait for eventual consistency, and report the wait.** A hosted index can keep returning a
   vector for a while after a successful delete. Probing immediately would report a residue that
   is really propagation delay. Suppress and reclaim both settle before returning and put the
   elapsed time in the measurement.

**On testing an adapter you cannot reach.** The Pinecone adapter is exercised against an
in-process fake (`tests/_pinecone_fake.py`) that reproduces the client's keyword-only signatures
and its eventual consistency. That proves the adapter is self-consistent and that the settle
logic works; it is **not** evidence about the live service, because a fake written alongside the
adapter can only encode its author's beliefs. Until it has been run against a real index, treat
it as unverified against Pinecone and say so wherever it is described.

## Registering the adapter

1. Add a `kind` to `StoreKind` in `src/tombstone/config.py` and its required keys in
   `StoreConfig._per_kind`.
2. Construct it in `Runtime.build_store` (`src/tombstone/registry.py`).
3. If the store holds vectors, subclass `VectorBackendBase` (`src/tombstone/stores/_vector.py`)
   and implement the eight storage primitives; the protocol methods, suppression-aware query
   paths, and the byte-scan probe come for free.

## The test template

```python
from tests.test_adapter_suite import adapter_contract

def test_my_store(tmp_path):
    store = MyStore("mine", tmp_path / "data")
    refs = [...]  # ArtifactRef per record you will write
    adapter_contract(store, lambda r: store.put(r.store_key, ...), refs)
```

The contract asserts: probe power on a present record; suppression hides without removing
bytes and is idempotent; reclaim removes bytes and a second reclaim is `noop`; other records
survive; `count()` and `sample_keys()` agree with what was written. The vector backends in this
repo run the same contract in `tests/test_adapter_suite.py`.
