# The lineage model

Deletion is a graph problem, not a search problem. If you know the lineage graph — source →
chunk → embedding → cache entry → training example → adapter — then erasure is a reachability
query followed by a fan-out, and verification is a per-node probe. If you do not know the graph,
you are guessing with semantic search, and semantic search both misses things and over-deletes
other people's data. Tombstone therefore never *selects* targets by similarity (Hard Rule 3);
similarity is used only to *verify* absence afterwards.

## Nodes

Every artifact is a node with exactly one `kind`, one `store`, the key the store knows it by,
a scope (tenant), a content hash (never the content), an optional embedding fingerprint (the
first 32 float32 values of the vector, hex — the byte pattern a physical probe looks for), and
the HMAC of the subject it descends from.

| kind | what it is | store examples | store_key |
|---|---|---|---|
| `source` | the original document | `docs` (the app's docstore) | the SOURCE artifact id |
| `chunk` | a split of a source, the unit of retrieval | `docs` | the CHUNK artifact id |
| `embed` | one vector in one index | `chroma:kb-v2`, `pgvector:kb-v1`, `faiss:kb`, `qdrant:kb` | the vector id in that index |
| `cache` | a cached answer (exact or semantic) | `exact-cache`, `semantic-cache` | `<cache key>@<subject16>` |
| `train` | one example in a fine-tune manifest | `ft-dataset` | the example id |
| `adapter` | a LoRA adapter (per shard, or unsharded) | `lora/support-v3` | `shard-07`, `unsharded` |
| `memory` | an agent-memory entry (Phase 8, optional) | `memory` | the entry key |

Artifact ids are ULIDs. Three of them are *derived* deterministically so that re-ingestion is
idempotent and LangChain's `index()` stays incremental: a SOURCE id from (scope, subject,
hashed source id, position); a CHUNK id from (source id, content hash); a TRAIN id from
(dataset, chunk id, shard). Nothing personal reaches a node: the subject id is HMAC-SHA256
under a per-installation pepper, the source id is SHA-256 hashed, and content appears only as
its hash.

## Edges

An edge is `parent → child` with a `via` tag naming the derivation:

| via | parent → child | written by |
|---|---|---|
| `chunk` | source → chunk | the vector-store wrapper on `add_documents` |
| `embed:<model>` | chunk → embed | the capture hook of each index (one edge per index) |
| `cache:exact` / `cache:semantic` | chunk → cache | the cache wrappers (parents parsed from the prompt's delimiter / the retrieved chunk ids) |
| `train:shard-N` | chunk → train | the dataset builder |
| `adapter:<key>` | train → adapter | adapter registration after training |
| `derived_from` | source → source | `stamp(..., derived_from=<id>)` for summaries, translations |
| `memory` | source → memory | the LangGraph memory adapter |

Multi-index is native: the same chunk added to two indexes is one CHUNK node with two EMBED
children. A chunk that two subjects' sources both produce (boilerplate) is one node reachable
from both; the trace flags it `shared`.

`mentions` is not an edge between artifacts. It is a separate record `SOURCE(other subject)
→ mentions → SUBJECT(this)`, written from what the *app* tells `stamp()` (Tombstone runs no
entity extraction). A trace returns those sources as `third_party_hits`, never as targets.

## What each store contributes

- **Vector indexes** contribute EMBED nodes with fingerprints and the `embed:<model>` edge; they
  also store the chunk text and the stamped metadata (including `tombstone.embed_id`), which is
  why a byte scan can look for the artifact's own id as well as its vector bytes.
- **The docstore** holds SOURCE and CHUNK rows keyed by artifact id.
- **Caches** contribute CACHE nodes; an entry that descends from chunks of several subjects gets
  one node per subject so each subject's trace reaches it.
- **The dataset** contributes TRAIN nodes sharded by a stable hash of the subject HMAC, so all
  of one subject's examples land in one shard — the precondition for exact unlearning.
- **Adapters** contribute one ADAPTER node per shard adapter (edges from that shard's examples)
  and one for an unsharded adapter (edges from every example).

## Append-only, and the tombstones table

`nodes` and `edges` are never updated or deleted. A deleted artifact gets a row in
`tombstones` (`artifact_id, tombstoned_seq, trace_id, reason`): `native-delete` when the app
called the store's own `delete()`, `erase:<reason>` when a saga suppressed it. The snapshot
hash a trace is computed over covers nodes, edges, mentions, registered stores and measured gaps
— deliberately not tombstones, so a saga's own suppression does not make its trace look stale.

## The trace, and what it refuses

`trace(subject, scope, snapshot)` is a pure function (`src/tombstone/lineage/trace.py`): same
inputs, byte-identical output, forever. It breadth-first walks parent → child from every SOURCE
the subject owns, in this scope only.

It raises `ScopeViolation` on any edge whose endpoints differ in scope, before anything is
deleted (Hard Rule 8). It raises `LineageGapError` when the subject has no lineage at all — it
never returns "nothing to delete" (Hard Rule 4). And it reports `gaps` for: a node stamped for
the subject that no SOURCE reaches (a missing parent edge); a store whose entries have no
lineage node (data ingested before Tombstone, or written around the wrapper — measured by
sampling the store's keys); an edge pointing at a node that does not exist. Non-empty gaps make
`erase` refuse unless `--accept-gaps` is passed, and then the receipt records
`UNVERIFIED(lineage-gap)` for the named stores.

## Numbers from a committed run

The fixture corpus in `tests/fixtures/corpus_small` (50 documents, 8 subjects) traced for
`S-0003` after ingestion into two indexes, a RAG call with both caches, and a dataset build
yields 5 sources, their chunks, twice as many embeds as chunks, one train row per chunk and at
least one cache node with no gaps; bypassing capture for 20 documents in one store yields a gap
naming that store (`tests/test_phase1_gate.py`). The bench corpus manifest
(`bench/corpus/manifest.json`) lists 3,115 documents: 2,000 AG News passages and 200 synthetic
subjects with 200 canaries and 81 cross-subject mentions.
