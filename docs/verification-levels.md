# Verification levels

This document is the product. For each backend and each level it states what a `VERIFIED` means,
what it does not, and why a managed database and a self-hosted one produce visibly different
receipts. Every artifact in a receipt carries exactly one of `VERIFIED(level)`,
`UNVERIFIED(reason)`, `RESIDUAL(measurement)`, `OUT_OF_SCOPE(reason)`, `NEEDS_HUMAN(reason)`.
A receipt is a record of what was done and checked. It is not a legal instrument.

## The four levels

| level | the question | the probe | what a pass proves | what it does not prove |
|---|---|---|---|---|
| logical | can the store still return it? | id lookup; metadata filter on the artifact id; top-40 and MMR with probe vectors | no API path this wrapper knows returns the artifact | bytes may still be on disk; other API paths outside the wrapper are not covered |
| physical | are the bytes still in storage? | byte scan of the persisted files for the vector fingerprint and the artifact id; dead-tuple count + heap/index/TOAST scan on Postgres; hash scan of dataset rows | the patterns are absent from every file the process can read | copies outside those files (backups, WAL, replicas, provider logs) — always `OUT_OF_SCOPE` |
| semantic | did the neighbourhood keep a trace? | *Ghost Echoes* Top-5 centroid drift vs a same-cluster control, bootstrap CI over the query budget | reported only: drift at control level or above it | it is a measurement of the neighbourhood, never proof the content is present or absent; never blocks |
| model | can the weights reproduce it? | greedy extraction from the subject's prefixes; loss and Min-K% membership inference vs a held-out set with a bootstrap CI | canaries not extractable and MIA at chance *against these two attacks* | unlearning verification is fragile (arXiv 2408.00929): stronger attacks may still separate members |

## The status lattice, in order

Evaluated top to bottom, first match wins (`src/tombstone/verify/levels.py`):

1. `scope_violation` — raise. 2. `not_suppressed` — raise (Hard Rule 5): nothing is assigned a
status without a durable suppression record. Then `dlq` → `UNVERIFIED`, `no_adapter` →
`UNVERIFIED`, `lineage_gap` → `UNVERIFIED`. 3. `third_party` → `NEEDS_HUMAN`. 4. `logical_fail`
→ `RESIDUAL(logical)`. 5. `physical_unsupported` → `UNVERIFIED(physical)`. 6. `physical_fail` →
`RESIDUAL(physical)`. 7. `model_residual` → `RESIDUAL(model)`. 8. `semantic_residual` →
`RESIDUAL(semantic)`. 9. `verified` at the highest applicable level.

Rule 5 before rule 6 is the honesty of the tool: a store that cannot be physically checked shows
up as `UNVERIFIED`, never as `VERIFIED` by omission.

## The probe design

A logical probe needs a query that would find the artifact if it were still there. Tombstone
never holds the content, so at capture time the vector-store wrapper embeds three short queries
derived from the chunk — its first, middle and last eight words — and stores each as (SHA-256 of
the query, embedding) in the `probes` table of the lineage db. The probe set at verification
time is those vectors plus the artifact's own fingerprint padded with zeros (the first 32 of its
dims, a weak but free probe). Probe rows are purged for every artifact that ends `VERIFIED`.
This is a bounded leak by design: fragments' embeddings, not the content, and gone after use. The
independent verifier has no lineage db and therefore no probe table; it probes by id, by filter,
and by the padded fingerprint only, and says so.

Physical probes look for two kinds of pattern. The artifact's own id string is present in the
stored metadata of every record Tombstone wrote, so its absence means the record is gone. The
content pattern (the fingerprint's float32 bytes, or a row's content hash) is what byte-identical
copies from *other subjects* share — boilerplate paragraphs produce them. When lineage shows
other live artifacts with the same bytes, the saga records a baseline count before reclaim and
calls it residue only if the count did not drop; when it cannot attribute shared bytes it reports
`UNVERIFIED(duplicate content)`, never `VERIFIED`.

## Per backend

| backend | logical | physical probe | physical reclaim | managed case |
|---|---|---|---|---|
| Chroma (embedded) | metadata flag + filter on every wrapper query path | scan `chroma.sqlite3` (queue, metadata, free pages) and every HNSW segment file for the fingerprint and the id | delete → rewrite the collection from survivors into a fresh segment → purge the embeddings queue → `VACUUM` → remove orphan segment dirs | persist dir not readable → `{LOGICAL}` |
| FAISS (`IndexHNSWFlat` in `IndexIDMap2`) | persisted exclusion set applied through `IDSelector` and post-filter | scan the index file and the sidecars; HNSW cannot `remove_ids`, so a native delete only drops the mapping | rebuild the index from survivors, atomic file replace | index file not writable → `{LOGICAL}` |
| Qdrant (local mode) | payload flag + `must_not` filter + post-filter | scan `storage.sqlite`: vectors are pickled as `BINFLOAT` opcodes (big-endian float64 derived exactly from the float32 values) | delete → rewrite the collection from survivors → `VACUUM` every sqlite file | server without snapshot API → `{LOGICAL}` |
| pgvector | `tombstoned` column in the `WHERE` clause + post-filter | `pgstattuple` dead-tuple count (when installed) + `pg_read_binary_file` over the heap, its TOAST table and the HNSW index (all segments), after a `CHECKPOINT` when superuser | `DELETE` + `REINDEX INDEX` + `VACUUM FULL` (or `VACUUM` + `REINDEX` when FULL is denied — recorded); credit to `vector-forget` for the recipe | a role without `pg_read_binary_file` → `{LOGICAL}`; a non-owner cannot `VACUUM`: receipt says `UNVERIFIED-managed` with the exact grant to ask for |
| docstore (SQLite) | `suppressed` column | scan the file for the row's artifact id | `DELETE` + `VACUUM` | — |
| exact cache (LangChain `SQLiteCache`) | row deleted at suppression (caches have no hide state) | scan for the cache key | `VACUUM` | — |
| semantic cache | entry (and optionally its k nearest neighbours) deleted at suppression | the backing backend's probe | the backing backend's reclaim | as the backing backend |
| dataset manifest | example marked `suppressed` (the trainer skips it) | re-hash every row of every shard file; the example id and content hash must be absent | drop the row, rewrite the shard file, re-hash the manifest, re-pin | — |
| LoRA adapter (sharded) | the serving ensemble is recomposed without the shard (immediate, reversible) | the shard's weights hash differs from the original and no copy remains | retrain the shard on its remaining examples, recompose | unsharded adapter: cannot be hidden; approximate unlearning is `RESIDUAL`-prone by construction |

## What the levels looked like on this machine

From `tests/test_verify.py` (Homebrew Postgres 16 with pgvector 0.8.1 built from source, Chroma
1.5.9, FAISS 1.15.0, Qdrant local mode): after the store's own `delete()`, the physical probe
found the deleted vector's bytes on all four backends — in `chroma.sqlite3`, in the FAISS index
file, in Qdrant's `storage.sqlite`, and in the pgvector HNSW index relation — and found nothing
after Tombstone's reclaim. The numbers over 200 subjects are in `RESULTS.md`.
