# Threat model: what breaks this

Tombstone erases what it saw arrive, from the stores it was given, to the level each store
permits, and says so. Everything below is a way for a subject's data to survive an erasure that
the receipt calls `VERIFIED`. The measured rates live in `RESULTS.md` under "Attacks that work
against Tombstone" (`bench/adversarial/run_attacks.py`); this page explains why each one exists
and what the operator must do about it. None of these is softened.

## The weakest links, in order

### 1. Lineage gaps — data ingested before capture (7.1)

Tombstone can only erase what it saw arrive. Documents that reached an index before the wrapper
was installed, or through a code path that bypasses it, have no lineage node. A trace for a
subject whose old documents are among them reaches only the new ones. What Tombstone does: gap
detection samples every registered store's keys and looks them up; any unlineaged entry makes
`erase` refuse unless `--accept-gaps` is passed, and then the receipt records
`UNVERIFIED(lineage-gap)` for the store — never `VERIFIED`. What the operator must do: re-ingest
legacy data through the wrapper (the stamps are deterministic, so `index()` treats unchanged
documents as skipped and only the lineage is backfilled).

### 2. Third-party mentions (7.4) — the unsolvable one

A subject named inside another subject's document is that other subject's data. Tombstone
records `mentions` edges only when the app supplies them at `stamp()` time and lists those
sources as `NEEDS_HUMAN`; it never searches for the subject by name or by similarity. Searching
would be worse: names collide, embeddings of "Harriet Vane" and "Harriet Vane's neighbour" are
close, and deleting or editing another subject's record on that basis is itself a violation of
that subject's rights and of the operator's own retention duties. The honest answer is a list
for a human, and a non-zero survival rate by construction. What the operator must do: run
entity extraction in the ingestion pipeline and pass `mentions=[...]`; review the `NEEDS_HUMAN`
rows.

### 3. Semantic residue (7.5) — the layer we cannot close

Correct deletion in a proximity-graph index leaves a measurable trace: the deleted item's
insertion-time routing decisions persist in the neighbours' geometry, and a full rebuild does not
remove it (*Ghost Echoes*, arXiv 2608.20352). Tombstone reproduces the measurement on its own
corpus, reports drift against a same-cluster control with a CI, and marks the artifact
`RESIDUAL(semantic)` when the CI excludes the control. It never blocks on it and never claims to
fix it.

Measured on our corpus (`bench/results/attacks-latest.json`, strategy 7.5, n=40 subjects): after a
full Tombstone erasure — suppress, reclaim, verify — an attacker who calibrates a threshold on
known subjects still separates erased from never-present at **70.0%** with a query budget of 5,
and **72.5%** at a budget of 10, against a 50% coin. The paper reports 61.1%; we reproduce the
effect rather than refute it. This is the one layer we publish as unfixable: no erasure Tombstone
performs moves this number, because the trace is in the surviving neighbours' geometry, not in
anything the deleted record still owns.

### 3b. A receipt verified against its own key proves integrity, not origin

A receipt carries the Ed25519 public key that signed it, and the signature cannot cover that key
— the signature is over the payload, and the key is what checks the signature. So
`tombstone verify --receipt r.json` *without* `--public-key` answers a narrower question than it
looks: were these bytes altered after signing? Anyone can write a receipt, sign it with a key
they generated a second ago, and it verifies. We demonstrated exactly that against our own
verifier, including a forged receipt whose notes read "erasure complete".

Two things close it, and the verifier now distinguishes them in its output rather than printing
`OK` for both:

- **Pass the operator's key.** `--public-key .tombstone/keys/public.pem` makes it a statement
  about origin, and a receipt signed by any other key is rejected with that reason. Without it the
  output says `signature: ed25519 self-asserted` and explains what that does and does not mean.
- **Pass the ledger.** Every receipt in one installation's ledger should be signed by that
  installation's key; one that is not was signed by someone else, whatever its own signature says.
  `--ledger` now fails on that mismatch as well as on a broken hash chain.

Neither helps if the attacker owns the machine and rewrites the ledger and the keypair together.
Against that, the receipt is evidence only to the extent the *verifier's* copy of the public key
came from somewhere the attacker does not control.

### 4. Backups, replicas, WAL, provider-side logs (7.7)

A filesystem snapshot taken before the erasure holds every vector; so does a `pg_dump`; so may
a replica's WAL and an embedding provider's request log. Every receipt lists these as
`OUT_OF_SCOPE`, and a receipt with an empty `out_of_scope` list cannot be constructed. The
experiment shows the declaration is not decorative: the subject is fully recoverable from the
snapshot after a `VERIFIED` erasure. What the operator must do: retention policy for backups,
and a second erasure after any restore.

### 5. Derived content without an edge (7.2)

A summary, translation or extraction the app stores as a new document carries the subject's
content with no path back to the source. Tombstone cannot see it. Mitigation: stamp derived
documents with `derived_from=<source artifact id>`; the trace then follows the edge. Measured
with and without.

### 6. Paraphrased cache hits (7.3)

A cached answer generated for a neighbour's question can quote the subject's chunk without that
chunk being among the cache entry's recorded parents. After erasure, a paraphrase of the original
question can still hit it. Mitigation: `SemanticCache(purge_k=k)` invalidates the k nearest cache
neighbours of every erased entry; its collateral cost (unrelated entries purged) is measured.

### 7. Approximate unlearning and relearning (7.6)

NPO and gradient-difference leave residual extractability at some rate and can be reversed by
light continued training on unrelated data. Exact shard retraining avoids this only if the
dataset was sharded by subject *before* training — the weights never saw the data. If it was
not, the honest options are a full retrain or a `RESIDUAL` receipt.

### 8. Races and concurrency (7.8, 7.9)

Between the CLI call and the durable suppression record, a concurrent reader can still get the
subject's chunk: that window is the suppression latency, measured under 50 concurrent readers.
Two subjects sharing a chunk erased concurrently must end in exactly the set difference; the
journal lock serialises writes and a clean failure is preferred to an interleaved journal.

## What the receipt can and cannot say

- `VERIFIED(physical)` means: the artifact's own record bytes are absent from every file this
  process could read, and its content bytes are absent or attributable to other subjects'
  byte-identical copies. It does not speak about files it could not read.
- `UNVERIFIED-managed` means: the store could not be checked at that level from this role. It
  names the grant to ask for.
- A receipt is a record of what was done and checked. It is not a legal instrument, and the
  vocabulary of assurance and regulatory compliance does not appear in any output of this tool;
  a CI check fails the build on it.
