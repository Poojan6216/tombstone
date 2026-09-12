# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `tombstone scan`: point it at a store this tool has never touched and it reports what a
  deletion request could and could not establish there today — entry count, how much of a sample
  carries a Tombstone stamp, whether byte-level proof is possible at all, and how much of the
  data is byte-identical to something else and therefore unattributable even with full lineage.
  No config, no lineage, no integration. Every other command needs lineage, and lineage only runs
  forward; this is the one that gives an answer to somebody who has the problem now.
  It issues no insert, update or delete, and it does not claim more than that: opening a store is
  enough to make some engines write to their own files, so the scan takes a census before and
  after and reports what moved. `ChromaStore(read_only=True)` skips the segment scrub for the
  same reason — harmless as that write is, a command that says it only looks has to only look.

## [0.1.0] - 2026-09-11

First release. Built end to end against a written specification, with every measured number
generated from a committed command into `RESULTS.md`.

### Added
- Lineage capture: `stamp()` for dicts, LangChain `Document`s and lists; capture hooks for
  Chroma, FAISS, Qdrant (local and server) and pgvector; `TombstoneVectorStore` for LangChain's
  `index()` / `RecordManager`; exact (LangChain `SQLiteCache`) and semantic caches with parent
  edges; sharded fine-tune dataset manifests; LangGraph-style memory entries.
- `trace`: a pure, deterministic reachability query with scope isolation, third-party mention
  listing, gap detection and pins.
- `erase`: two-phase suppression → reclaim → verify as a journaled, resumable saga with a
  dead-letter queue; per-backend physical reclaim (pgvector `REINDEX + VACUUM FULL` credited to
  vector-forget; Chroma segment rewrite + queue purge + `VACUUM`; FAISS rebuild; Qdrant rewrite).
- Verification at four levels: logical probes (probe table of derived-query embeddings),
  physical byte scans (float32 fingerprints, pickle `BINFLOAT` patterns, Postgres heap/index/TOAST
  reads, dataset hash scans), *Ghost Echoes* drift vs a same-cluster control, and model probes
  (canary extraction, loss and Min-K% membership inference with bootstrap CIs).
- The model as an artifact: per-shard LoRA adapters served as a likelihood-weighted SISA
  prediction ensemble; exact shard retraining; NPO and gradient-difference approximate
  unlearning; relearning attack.
- Receipts: Ed25519-signed, hash-chained ledger, `tombstone replay` from the journal, and an
  independent verifier that needs neither the lineage db nor the pepper.
- MCP server (mcp 2.x): `tombstone.trace/verify/erase/receipt/status`; `erase` confirms through
  elicitation with a sealed request state and refuses on clients without it.
- Benchmarks: residue matrix (B0–B4 × four backends), unlearning matrix (M0–M4), nine attack
  strategies, plots, and a generated `RESULTS.md`.
- `tombstone forget <subject> --reason <id>`: trace, show what was found, ask once, erase. The
  two-command path made a person copy a 26-character trace id between commands; the confirmation
  is now a human answering a list of real artifacts instead of a `--confirm` flag.
- A public Python API — `from tombstone import trace, forget` — so an erasure can run inside the
  operator's own delete-account handler with no command at all. Both are resolved lazily, so
  `import tombstone` still pulls in nothing heavy.
- `tombstone.forget` on the MCP server: one tool call from a subject id, with the same
  elicitation confirmation `tombstone.erase` requires, and it erases exactly the set that
  confirmation listed rather than re-tracing afterwards.
- `tombstone ui`: a local page for the people who receive deletion requests and do not use a
  terminal. Loopback only, no cookies, and every API call must carry a per-run token sent in a
  request header, so a page the operator is browsing cannot drive their deletion tool. It
  refuses lineage gaps, requires a reason, and will not erase against a view that has gone stale.

### Fixed
- The test helper that gives each test its own database built the new DSN by splitting the old
  one on its last "/". For a server reached over a unix socket, whose socket directory rides in a
  `?host=` query parameter, that lands inside the path and appends the database name to the
  *host* — so the connection hunts for a socket in a directory that does not exist and reports
  "is the server running?" about a server that is running perfectly well. It now swaps only the
  database component. This is what failed the first v0.1.0 release build.
- `release.yml` ran the full suite with no Postgres service, so the release gate tested a
  different environment than CI did. It now gets the same pgvector service and DSN, and a repo
  check asserts the two stay in step.
- The MCP server reports its version in `serverInfo`, which was an empty string.
- The receipt labelled byte-identical duplicate content `UNVERIFIED-managed` and told the
  operator to grant file access or run `VACUUM FULL` — a remedy that cannot help, because the
  store was read perfectly well and the remaining bytes belong to other live records. That case
  now reads `UNVERIFIED-duplicate` and says there is nothing to fix. The rule id, outcome and
  level are unchanged, so replay and independent verification are unaffected.
- Chroma: the physical probe on Linux CI found an erased vector in the rewritten segment's
  `data_level0.bin` after a clean reclaim. chroma-hnswlib persists the index at its allocated
  capacity from a `malloc`'d buffer it never clears, and does so on every open until the index
  reaches `sync_threshold`, so the slots past `cur_element_count` carry whatever the allocator
  handed over — on glibc, the buffer the deleted collection's index had just freed. The adapter
  now zeroes those slots after the rewrite, when a new segment directory appears, and at open;
  the receipt's reclaim measurement records `unused_slot_bytes_zeroed`. A header the decoder does
  not recognise is left alone and counted in `segments_not_scrubbed`.

### Decisions worth knowing
- Intel macOS pins torch to 2.2.2 (and transformers < 5, numpy < 2) via platform-marked
  constraints; everything else resolves normally.
- Weight-level merges of shard adapters lost memorised facts in measurement, so serving is a
  prediction-level ensemble.
- Byte-identical content across subjects (boilerplate) is decided on the lineage graph alone:
  if another *live* artifact holds the same content, no scan can say whose copy it found, and the
  receipt says `UNVERIFIED(duplicate content)`. Every version of this rule that compared byte
  counts before and after was timing-sensitive — a background segment flush moves the number —
  and a saga killed and resumed mid-flight could reach a different verdict than a clean run on
  identical data. A fact about the graph cannot move underneath a probe; a byte count can.
