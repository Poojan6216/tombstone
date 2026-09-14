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
```text
$ tombstone verify --subject S-0042 --after-native-delete
after native delete()
subject: hmac:f675…1d04   (raw id never logged)
artifacts descending from this subject: 23

  source  ×3 docs                   HIDDEN     logical PASS, physical FAIL
                                               bytes for 3/3 ids located in docs.sqlite
  chunk   ×4 docs                   HIDDEN     logical PASS, physical FAIL
                                               bytes for 4/4 ids located in docs.sqlite
  embed   ×4 chroma:kb-v2           HIDDEN     logical PASS, physical FAIL
                                               bytes for 4/4 ids located in chroma.sqlite3 + data_level0.bin
  embed   ×4 pgvector:kb-v1         HIDDEN     logical PASS, physical FAIL
                                               bytes for 4/4 ids located in documents_embedding_hnsw + pg_toast.pg_toast_17061
  cache   ×1 exact-cache            PRESENT    cached answer still returned
  cache   ×1 semantic-cache         PRESENT    cached answer still returned
  train   ×4 ft-dataset             PRESENT    still retrievable by id
  adapter ×2 lora/support-v3        GONE       no extraction prompts

verdict: NOT ERASED.  21/23 artifacts still hold recoverable content.
exit 2
```
<!-- generated:demo1:end -->

The deletion was complete at exactly one layer. The full three demos, including the receipt that
refuses to say "complete", are in [docs/demo.md](docs/demo.md).

## Headline numbers

Every number below is produced by a committed command from a committed JSON file
(`bench/results/`). `RESULTS.md` is generated, never hand-edited.

<!-- generated:headline:start -->
<!-- measured:start -->
1. **After a native `delete()`, 92.0%-100.0% of a subject's vectors are still physically recoverable** from the index files, depending on the backend (chroma 99.8%, faiss 92.0%, qdrant 100.0%, pgvector 94.9%; 200 subjects each) — while every one of them is logically gone (100.0% exclusion). After `tombstone erase`, 0.0% of the subjects' own records remain and 0.0% of their vectors have bytes still attributable to them. A further 0.0%-8.0% have bytes that are byte-identical to records belonging to *other* subjects (shared boilerplate); a byte scan cannot tell those two copies apart, so they are reported separately and never counted as the erased record's residue. Source: `bench/results/residue-latest.json`, command `uv run python bench/residue/run_residue.py --all`.
2. **Exact shard unlearning vs approximate** on `Qwen/Qwen2.5-0.5B` (20 subjects): before, canary extraction 17/20, MIA AUC 1.00 [1.00,1.00], held-out perplexity 298.7; exact shard retrain: canary extraction 0/20, MIA AUC 0.41 [0.23,0.61], held-out perplexity 231.0; NPO: canary extraction 0/20, MIA AUC 0.69 [0.51,0.84], held-out perplexity 82.4; gradient difference: canary extraction 5/20, MIA AUC 0.91 [0.76,1.00], held-out perplexity 103.5. Source: `bench/results/unlearn-latest.json`, command `uv run python bench/unlearn/run_unlearn.py --all`.
3. **The layer nobody can erase**: after a full Tombstone erasure, an attacker estimating "was this subject ever here?" from retrieval-context drift (*Ghost Echoes* protocol) reaches paired-comparison accuracy 70.0% at budget 5, 72.5% at budget 10, 65.0% at budget 20, 70.0% at budget 40. Tombstone measures and reports this; it does not fix it. Source: `bench/results/attacks-latest.json`.
<!-- measured:end -->
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
  says `UNVERIFIED-managed`, and that is the correct answer. The Pinecone adapter is the clearest
  case: it traces and suppresses exactly as the others do, and states plainly that the stored
  bytes were never examined, because the service exposes no way to examine them. It has been
  tested against an in-process fake of the client, **not against a live Pinecone index**.
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
uv pip install 'tombstone-erase[chroma,faiss,qdrant,pgvector,pinecone,langchain]'   # store adapters
uv pip install 'tombstone-erase[train]'    # the model leg (torch, transformers, peft)
uv pip install 'tombstone-erase[mcp]'      # the MCP server
```

`pip install` works the same way. To run against unreleased main instead:

```bash
uv tool install git+https://github.com/Poojan6216/tombstone
```

The core install pulls only `pydantic`, `cryptography` and `pyyaml` — every backend, the training
leg and the MCP server are extras, so `import tombstone` stays fast and drags in nothing heavy.

**Needs Python 3.11 or newer.** On anything older, `pip` does not say so plainly — it reports
`Could not find a version that satisfies the requirement tombstone-erase (from versions: none)`,
which reads as "this package does not exist" when it means "none of its releases run on your
Python". The real message is the line above it, which scrolls past: `Ignored the following
versions that require a different python version`. `uv tool install` avoids this by fetching a
suitable interpreter itself; with pip, point it at one:

```bash
python3.11 -m pip install tombstone-erase
```

**The PyPI name is `tombstone-erase`, not `tombstone`.** `pip install tombstone` fetches an
unrelated project (a directory-watching utility). Both install a top-level `tombstone` module, so
an environment holding the two ends up with whichever was installed last and **pip warns about
none of it** — install this one second and it works; install it first and `tombstone --version`
starts raising `ImportError` with both still listed in `pip list`. If that happens:

```bash
pip list | grep -i tombstone     # if plain `tombstone` is there, it is not this project
pip uninstall tombstone && pip install --force-reinstall tombstone-erase
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

### Already have a database full of vectors?

Everything else here needs lineage, and lineage only runs forward. If you have three years of
embeddings already in Chroma and a deletion request in your inbox this morning, start with the
one command that needs no config, no integration and no prior installation:

```bash
tombstone scan            # or: tombstone scan /path/to/your/store
```

```text
  chroma:customer-kb   ./chroma_db
    entries            47,203
    tombstone stamps   0/200 sampled (0%)  — nothing here arrived through Tombstone
    shared fingerprints 16/200 sampled (8%) — byte-identical to another entry
    byte-level proof   available

If a deletion request arrived today:
  ✗ nothing here can be traced to a person by this tool.
  ✓ if your application records which ids belong to whom, you can still delete by id
  ! some entries are byte-identical to others, so no scan could attribute them
    to one person even with full lineage. That is a fact about the data.
```

It issues no insert, update or delete. It does not claim your files are untouched, because
opening a store is enough to make some engines write to their own files — Chroma rewrites its
index header on every open, by any client — so the scan takes a census before and after and
tells you exactly what moved.

Then, when a deletion request arrives:

```bash
tombstone forget S-0417 --reason dsr-2026-0912    # traces, shows what it found, asks once, erases
```

```text
47 things exist because of subject hmac:99d5…83ec   (raw id never stored)

  chroma:kb-v2                     10
  docs                             15
  exact-cache                       1
  faiss:kb-v1                      10
  ft-dataset                       10
  semantic-cache                    1

  kinds: cache×2, chunk×10, embed×20, source×5, train×10
  lineage gaps: none

erase all 47 of these? this cannot be undone  [y/N]
```

The pieces are still there when you want them, and everything else the tool does:

```bash
tombstone trace --subject S-0417                  # look without touching anything
tombstone erase --trace <trace-id> --reason dsr-2026-0912 --confirm
tombstone receipt
tombstone verify --receipt .tombstone/receipts/<id>.json --public-key .tombstone/keys/ed25519.pub   # independent check
tombstone replay                                  # re-derive every receipt from the journal
tombstone ui                                      # a local page for whoever handles the request
tombstone mcp                                     # MCP server; erase requires an elicitation confirmation
```

### From your own code

A deletion request arrives in your product, not in a terminal — a customer clicks "delete my
account", a ticket lands in a queue. So the erasure belongs in the handler you already have:

```python
from tombstone import trace, forget

held = trace("S-0417")                       # what do we hold? reads only
print(held.count, held.by_store, held.gaps)

result = forget("S-0417", reason=f"dsr-{ticket_id}")    # destructive
if not result.ok:
    alert_privacy_team(result.report)        # something could not be confirmed; it says what
```

Calling `forget` is the confirmation — a call written in your own source is already a deliberate
act, and a boolean people paste without reading protects nobody. It returns the same receipt the
CLI writes, `tombstone replay` re-derives it identically, and it raises rather than reporting
"nothing to delete" for a subject it has no lineage for. Both functions resolve lazily, so
`import tombstone` still pulls in nothing heavy.

### For whoever actually handles the request

```bash
tombstone ui        # http://127.0.0.1:7878 — search a person, see what is held, erase, read the receipt
```

Support and legal receive the DSR and do not use a terminal. The page binds loopback only, and
every request to it must carry a per-run token that is printed with the URL, so a page the
operator happens to be browsing cannot drive their deletion tool. It refuses lineage gaps, asks
for a reason and writes the same journal and receipt as the CLI.

### Driving it from an AI assistant (MCP)

`tombstone mcp` exposes six tools — `tombstone.forget`, `tombstone.trace`, `tombstone.verify`,
`tombstone.erase`, `tombstone.receipt`, `tombstone.status` — over stdio or streamable HTTP. Point
any MCP client at it:

```json
{
  "mcpServers": {
    "tombstone": {
      "command": "tombstone",
      "args": ["mcp", "--config", "/absolute/path/to/tombstone.yaml"]
    }
  }
}
```

The assistant can then trace a subject, inspect a receipt, or ask for coverage, and
`tombstone.forget` does the whole thing from a subject id in one call. **Neither `forget` nor
`erase` can execute without an explicit confirmation** carried in the request, on every protocol
revision, and `forget` erases exactly the set that confirmation listed. An agent cannot talk its
way into deleting your data.

One thing this does not remove: Tombstone can only trace what it saw arrive. If the app was not
wrapped at ingest, `trace` reports a lineage gap and refuses to claim an erasure it cannot back
up — see [the attacks](RESULTS.md#attacks-that-work-against-tombstone), 7.1.

## Results

<!-- generated:results:start -->
<!-- measured:start -->
| backend | method | logical exclusion | physical residue | Recall@5 before → after | wall/erasure |
|---|---|---|---|---|---|
| chroma | B0 native delete() | 100.0% | 99.8% | 98.5% → 98.0% | 0.03s |
| chroma | B1 delete + vendor compact | 100.0% | 100.0% | 98.5% → 98.5% | 0.03s |
| chroma | B2 delete + full rebuild | 100.0% | 78.6% | 98.5% → 98.5% | 6.74s |
| chroma | B3 Tombstone suppress only | 100.0% | 100.0% | 98.5% → 98.0% | 11.41s |
| chroma | B4 Tombstone full | 100.0% | 0.0% | 98.5% → 98.5% | 20.98s |
| faiss | B0 native delete() | 100.0% | 92.0% | 94.0% → 94.0% | 0.05s |
| faiss | B1 delete + vendor compact | 100.0% | 92.0% | 94.0% → 94.0% | 0.04s |
| faiss | B2 delete + full rebuild | 100.0% | 0.0% | 94.0% → 98.5% | 0.46s |
| faiss | B3 Tombstone suppress only | 100.0% | 100.0% | 94.0% → 94.0% | 8.99s |
| faiss | B4 Tombstone full | 100.0% | 0.0% | 94.0% → 98.5% | 11.22s |
| qdrant | B0 native delete() | 100.0% | 100.0% | 98.5% → 98.5% | 0.01s |
| qdrant | B1 delete + vendor compact | 100.0% | 100.0% | 98.5% → 98.5% | 0.01s |
| qdrant | B2 delete + full rebuild | 100.0% | 0.0% | 98.5% → 98.5% | 6.16s |
| qdrant | B3 Tombstone suppress only | 100.0% | 100.0% | 98.5% → 98.5% | 10.47s |
| qdrant | B4 Tombstone full | 100.0% | 0.0% | 98.5% → 98.5% | 18.92s |
| pgvector | B0 native delete() | 100.0% | 94.9% | 82.0% → 99.0% | 0.00s |
| pgvector | B1 delete + vendor compact | 100.0% | 93.2% | 89.5% → 99.0% | 0.04s |
| pgvector | B2 delete + full rebuild | 100.0% | 92.7% | 85.5% → 99.0% | 0.78s |
| pgvector | B3 Tombstone suppress only | 100.0% | 100.0% | 86.5% → 86.5% | 7.96s |
| pgvector | B4 Tombstone full | 100.0% | 0.0% | 88.5% → 99.0% | 10.81s |
<!-- measured:end -->
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

## Security

The core install carries no packages with open advisories; the optional extras do, and
[SECURITY.md](SECURITY.md) lists each one, what it affects, and the two we have not fixed and why.
Subject ids are HMACs under a per-installation pepper, and no content of any kind reaches the
lineage database, journal, ledger or receipts — enforced by a test that is never skipped.

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
no accounts. Never report a number you did not measure. Each one is enforced by a test in
`tests/` — see `tests/test_repo_checks.py`, `tests/test_secrets.py`, `tests/test_erase.py` and
`tests/test_verify.py`.

## License

Apache-2.0.
