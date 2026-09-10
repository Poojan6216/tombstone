# Tombstone

**`delete()` is a lie.** Tombstone tracks where a data subject's data actually went — chunks,
embeddings in several indexes, caches, training examples, fine-tuned adapter weights — erases it
everywhere, and hands back a receipt that says exactly what was proven gone, at which level, and
what was not.

It sits beside an existing LangChain / vector-store / fine-tuning pipeline. It selects nothing by
similarity: erasure is a reachability query over a lineage graph captured at ingest. It never
claims a verification level it did not achieve, and a receipt with no `OUT_OF_SCOPE` entries
does not exist.

**Start with what beats it:** [Attacks that work against Tombstone](RESULTS.md#attacks-that-work-against-tombstone)
— measured, with rates, including the layer nobody can erase.

![The lineage graph: source, chunk, embed, cache, train, adapter and memory artifacts, and the
four verification levels with what each proves and what it does not.](docs/lineage.svg)

## Demo 1 — what a native `delete()` reaches

A RAG app over the benchmark corpus with two indexes, a semantic cache, a fine-tune dataset and a
LoRA adapter. The operator deletes the way everyone deletes — `vectorstore.delete(ids=[...])` on
both indexes, source rows dropped — and believes the deletion is complete.

<!-- generated:demo1:start -->
_Run `uv run python bench/demo.py` to generate the demos._
<!-- generated:demo1:end -->

The deletion was complete at exactly one layer. The full three demos, including the receipt that
refuses to say "complete", are in [docs/demo.md](docs/demo.md).

## Headline numbers

Every number below is produced by a committed command from a committed JSON file
(`bench/results/`). `RESULTS.md` is generated, never hand-edited.

<!-- generated:headline:start -->
_Benchmarks have not been run yet; no numbers are claimed._
<!-- generated:headline:end -->

## Known limitations — read these first

- **Tombstone can only erase what it saw arrive.** Data ingested before capture was enabled is a
  lineage gap; the tool refuses to call it erased ([attacks](RESULTS.md#attacks-that-work-against-tombstone), 7.1).
- **Third-party mentions are not erased.** A subject named inside another subject's document is
  listed for human review, not deleted, because deleting it would be a different violation (7.4).
- **Semantic residue is not erasable at the application layer.** Correct deletion in a
  proximity-graph index leaves measurable retrieval drift, and rebuilds do not remove it
  (*Ghost Echoes*, arXiv 2608.20352). Tombstone measures and reports it; it does not claim to fix it (7.5).
- **Approximate unlearning is approximate.** NPO and gradient-difference leave residual
  extractability and can be reversed by light continued training. Exact shard retraining avoids
  this only if the dataset was sharded by subject *before* training (7.6).
- **Managed databases cannot be physically verified** without maintenance rights. The receipt
  says `UNVERIFIED-managed`, and that is the correct answer.
- **Backups, snapshots, replicas, WAL and provider-side logs are out of scope** and always listed
  as such (7.7).
- **A receipt checked against its own key proves it was not altered, not who wrote it.** The
  signing key travels inside the receipt, so verification without `--public-key` is integrity
  only; pass the operator's `.tombstone/keys/public.pem` (and `--ledger`) to make it a statement
  about origin. The verifier says which of the two you got ([threat model](docs/threat-model.md)).
- **A receipt is a record, not a legal instrument.** It states what was done and checked. Whether
  that satisfies a regulator is a question for a lawyer, and the tool says so in its output.
- **Derived content the app does not stamp is invisible** (7.2). **Single scope per erasure.**
  Four vector backends, two caches, one training framework, small models only in the benchmark.

## The mechanism, in one paragraph

Deletion is a graph problem, not a search problem. `stamp()` attaches a subject and a source to
every document at ingest; capture hooks on the vector-store wrapper, the caches and the dataset
builder propagate that to every derived artifact as nodes and edges in a small append-only
SQLite graph. `trace()` is a pure reachability query over a snapshot of that graph — same inputs,
byte-identical output, forever. `erase` is a journaled, resumable saga over the traced set: write
a tombstone marker that every retrieval path honours immediately, then physically reclaim per
store, then probe each artifact at the strongest level the store permits — logical (not
retrievable), physical (bytes not in storage), semantic (drift measured against a same-cluster
control), model (canaries not extractable, membership inference at chance). The receipt is
Ed25519-signed, hash-chained, replayable from the journal, and checkable by an independent
verifier with no access to the lineage database.

## Install

```bash
uv tool install tombstone-erase            # CLI: tombstone
uv pip install 'tombstone-erase[chroma,faiss,qdrant,pgvector,langchain]'   # store adapters
uv pip install 'tombstone-erase[train]'    # the model leg (torch, transformers, peft)
uv pip install 'tombstone-erase[mcp]'      # the MCP server
```

Ten-minute path (scripted in `tests/test_first_use.py`):

```bash
tombstone init                                   # detects LangChain, Chroma/FAISS/Qdrant dirs, PG_DSN, adapters; writes tombstone.yaml
```

```python
from tombstone.integrations.langchain import TombstoneVectorStore
from tombstone.lineage.stamp import stamp

vs = TombstoneVectorStore.from_config("chroma:kb-v2", embeddings)   # wrap the store in one line
docs = stamp(docs, subject_id="S-0417", source_id="crm/417.pdf", scope="default", pepper=pepper)
index(docs, record_manager, vs, cleanup="incremental", source_id_key="source")  # stock LangChain
```

```bash
tombstone trace --subject S-0417
tombstone erase --trace <trace-id> --reason dsr-2026-0912 --confirm
tombstone receipt
tombstone verify --receipt .tombstone/receipts/<id>.json --public-key .tombstone/keys/ed25519.pub   # independent check
tombstone replay                                  # re-derive every receipt from the journal
tombstone mcp                                     # MCP server; erase requires an elicitation confirmation
```

## Results

<!-- generated:results:start -->
<!-- generated:results:end -->

Full matrices, the hyperparameter grid, the anti-results and the attacks that beat Tombstone:
[RESULTS.md](RESULTS.md). Attacks first: [Attacks that work against Tombstone](RESULTS.md#attacks-that-work-against-tombstone).

## Docs

- [docs/lineage-model.md](docs/lineage-model.md) — the graph, the edges, what each store contributes
- [docs/verification-levels.md](docs/verification-levels.md) — per backend, per level, what is proven and what is not
- [docs/unlearning.md](docs/unlearning.md) — methods, metrics, and why the metrics are fragile
- [docs/threat-model.md](docs/threat-model.md) — what breaks this
- [docs/adapters.md](docs/adapters.md) — how to add a store in under 200 lines
- [docs/demo.md](docs/demo.md) — the three demos, generated
- [docs/writeup.md](docs/writeup.md) — the technical post

## Prior art — who is already here, and where they stop

- **Ghost Vectors** (arXiv 2606.18497) and **Ghost Echoes** (arXiv 2608.20352), Trinity College,
  2026: soft-deleted embeddings are physically recoverable and invertible; correct deletion still
  leaves measurable retrieval drift; rebuilds do not remove it. They are attacks and measurements,
  not a deletion system. Our residue benchmark reproduces their drift measurement on our corpus
  and credits their methodology; the measurement is theirs.
- **vector-forget** (PyPI): hard-delete plus physical-residue verification for pgvector
  (`pgstattuple`, `REINDEX + VACUUM`) — the right approach, which we use and credit. pgvector
  only; no lineage; explicitly no backups, replicas, WAL, caches, other stores or fine-tuned
  weights. We treat it as a collaborator.
- **forgetlayer** (PyPI): cascade-delete for agent memory (Mem0/Zep) with Ed25519 receipts and an
  independent verifier — "a store grading its own homework convinces no auditor", which is exactly
  right and is where our independent verifier comes from. Agent memory only; no lineage capture at
  ingest, no vector-store physical layer, no training layer.
- **sura-rag** (PyPI): delete → probe → guardrail → certify for RAG, local-only. No lineage,
  single store, and its wording over-claims relative to what a probe can show; that word does not
  appear in this repository.
- **LangChain's indexing API** (`index`, `RecordManager`): genuine source → chunk lineage and the
  substrate we build on. It calls the store's soft `delete()`, knows nothing about caches, second
  indexes or training data, has no subject dimension, and cannot verify anything.
- **Machine unlearning literature**: SISA (Bourtoule et al. 2021), NPO (arXiv 2404.05868), TOFU
  and MUSE, and the verification literature — *Verification of Machine Unlearning is Fragile*
  (arXiv 2408.00929), *Dual-View Inference Attack* (arXiv 2512.16126), SMI (arXiv 2602.01150) —
  whose main finding is that verification is easy to fool. That finding is why Phase 7 attacks our
  own unlearning verification.
- **Data-lineage tooling** (OpenLineage, DataHub, Atlas): dataset- and job-level lineage; none
  model chunk → embedding → cache → training-example edges or execute deletion.

Tombstone's contribution: the first installable tool that captures subject-level lineage across
the whole derived-artifact graph of an LLM application, executes erasure as a fan-out over that
graph rather than a search, verifies every artifact at the strongest layer the store permits and
reports the layer, and includes the fine-tuned model as a first-class artifact with measured
unlearning, published together with the attacks that make that measurement unreliable.

## Hard rules the code is held to

No LLM in the decision or verification path. Never claim a level you did not achieve. Targets
come from lineage, never from similarity. Fail loud on missing lineage. Suppression before
reclaim. Resumable saga, never partial success. Never log raw personal data. Scope isolation.
Determinism and replay. Never delete what you did not trace. No telemetry, no hosted components,
no accounts. Never report a number you did not measure. Each has a test; see `BUILD_SPEC.md`.

## License

Apache-2.0.
