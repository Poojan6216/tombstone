# Here is every place your data still is after you delete it

*A technical post on Tombstone. Every number in this post comes from `RESULTS.md`, which is
generated from committed JSON by committed commands; the slots marked «…» are filled from the
benchmark run named in each section.*

## The one-layer deletion

A company builds a RAG assistant over its documents and fine-tunes a small model on some of them.
A customer invokes Article 17. Someone runs `delete()`. The source row is gone. Everybody moves
on.

We built the app, deleted the way everyone deletes — `vectorstore.delete(ids=[...])` on both
indexes, source rows dropped — and then asked the storage layer what it still held. The answer,
for one subject with «N» derived artifacts, was «K» still recoverable: the vector bytes of every
embedding in both index files, the cached answer that quotes the deleted chunk, the training row,
and a LoRA adapter that will complete the subject's membership number from a twenty-token prefix.
The deletion was complete at exactly one layer. (`docs/demo.md`, Demo 1; `bench/demo.py`.)

Across «S» subjects and four vector backends, a native delete left «P%» of a subject's vectors
physically recoverable from the index files while «L%» of them were logically gone
(`bench/results/residue-latest.json`, B0 row). That is not a bug in any one store. HNSW-backed
stores soft-delete by design, SQLite keeps freed pages until `VACUUM`, and a queue that logged the
insert keeps the bytes until it is purged. *Ghost Vectors* (arXiv 2606.18497) showed the same
thing on Weaviate, FAISS and Chroma and inverted the recovered vectors back to text; we reproduce
the recovery on Chroma, FAISS, Qdrant's local storage and pgvector's HNSW index.

## The receipt that refuses to say "complete"

The second demo is the one to lead with, and it is the honest one. The pgvector instance is
reached through a role without file-read or maintenance rights — a managed database. The adapter
was trained without sharding. The subject is named inside two other subjects' documents.

Tombstone erases what it can, verifies each artifact at the strongest level the store permits,
and writes a receipt that says `UNVERIFIED-managed` for the database it could not check,
`RESIDUAL(model)` for the adapter it could only approximately unlearn (canary «c/n» still
extractable, MIA AUC «a» [«lo», «hi»]), and `NEEDS_HUMAN 2` for the two documents it will not
touch because they belong to other people. Exit code 2. A tool that printed "erasure complete"
here would be lying to a regulator on the operator's behalf.

## Deletion is a graph problem

If you know the lineage graph — source → chunk → embedding → cache entry → training example →
adapter — erasure is a reachability query followed by a fan-out, and verification is a per-node
probe. If you do not know it, you are guessing with semantic search, which both misses things and
deletes other people's data. Tombstone captures the graph at ingest with a one-line wrapper around
the vector store and a `stamp()` on documents, never selects a target by similarity, and treats
`trace()` as a pure function: same subject, scope and snapshot, byte-identical output, forever.
Erasure is a journaled saga — suppress first (a marker every retrieval path honours before any
bytes move), reclaim per store, verify per artifact — that survives `SIGKILL` at «15» random
points and produces the same receipt on resume (`tests/test_erase.py`).

## Four verification levels, and what each one does not prove

*Logical*: no API path the wrapper knows returns the artifact. *Physical*: its bytes are absent
from every file the process can read — and only those files; backups, WAL, replicas and
provider-side logs are `OUT_OF_SCOPE` on every receipt, and a receipt with an empty
`OUT_OF_SCOPE` list cannot be constructed. *Semantic*: retrieval-context drift measured against
a same-cluster control, reported and never called proof. *Model*: canaries not extractable and
membership inference at chance against two attacks — with the literature's own warning that
verification of unlearning is fragile (arXiv 2408.00929).

The lattice evaluates in a fixed order and the honesty lives in one rule: "store cannot be
physically checked" is decided *before* "bytes not found", so a managed database can never come
out `VERIFIED` by omission. A second rule handles boilerplate: when other subjects hold
byte-identical chunks, a byte scan cannot tell whose copy it found, so the receipt says
`UNVERIFIED(duplicate content)` unless a pre-reclaim baseline shows the subject's own copy went
away.

## The model is an artifact too

Sharding the training set by subject before training makes exact unlearning cheap: drop the
subject's rows, retrain that shard's adapter from the base model, recompose. We measured the
composition before choosing it. Merging the shard adapters' LoRA weights — sum, concatenation
or average — lost the memorised facts (0/6, 0/6, 1/6 canaries where each shard alone recovered
5/6, `bench/results/composition-latest.json`), so the serving model is what SISA actually
prescribes: a prediction-level ensemble, weighted by how well each shard recognises the context.

On «Q» subjects with `Qwen2.5-0.5B` on a CPU: exact shard retrain took canary extraction from
«c0/n» to «c3/n» and MIA AUC from «a0» to «a3» [«lo3», «hi3»] at a held-out perplexity cost of
«dppl»; NPO left «c1/n» extractable at AUC «a1», gradient difference «c2/n» at «a2»
(`bench/results/unlearn-latest.json`). And the un-forgetting: «R» fine-tuning steps on unrelated
news text brought «c1r/n» of the NPO-forgotten canaries back. The retrained shards stayed at
«c3r/n». The data is not in the weights; nothing can be relearned from it.

## The layer nobody can erase

*Ghost Echoes* (arXiv 2608.20352) found that even a correct deletion leaves the deleted item's
insertion-time routing decisions in the proximity graph, measurable as Top-K centroid drift
against a same-cluster control, and that a full rebuild does not remove it. On our corpus, after a
full Tombstone erasure, an attacker estimating "was this subject ever here?" from drift reached
paired-comparison accuracy of «acc5» at a query budget of 5 and «acc40» at 40
(`bench/results/attacks-latest.json`, 7.5). Tombstone measures and reports this. It does not
claim to fix it, because nothing at the application layer can.

## The attacks that beat it

We attacked our own eraser and published the rates: data ingested before capture was enabled
(«r71» of subjects' canaries survive at 30% pre-capture; the refusal fires and the receipt says
`UNVERIFIED(lineage-gap)`), summaries stored without a `derived_from` edge («r72» survive; «r72m»
with the edge), a subject quoted in other subjects' documents with no `mentions` edge («r74» —
non-zero by construction, because searching for them would over-delete other people's data), a
filesystem snapshot taken before the erasure («r77» fully recoverable — that is what
`OUT_OF_SCOPE` means), paraphrased semantic-cache hits («r73», and the collateral cost of the
neighbourhood purge), a suppression window under 50 concurrent readers of «lat» ms, and the
relearning attack above. The full table is `RESULTS.md` § "Attacks that work against Tombstone".

## What this is not

A receipt is a record of what was done and checked. It is not a legal instrument, and Tombstone
never uses the vocabulary of assurance. It cannot erase what it did not see arrive, it will not
touch other subjects' data, it cannot close the semantic layer, and it cannot reach a backup. The
contribution is narrower than "GDPR for AI" and, we think, more useful: an installable tool that
knows where a subject's data went, erases it everywhere it can, measures the rest, and says so.

Credits: *Ghost Vectors* and *Ghost Echoes* (Trinity College) for the problem and the drift
protocol; `vector-forget` for the pgvector residue recipe; `forgetlayer` for the independent
verifier framing; LangChain's indexing API for the source-to-chunk substrate; Bourtoule et al.
for SISA; Zhang et al. for NPO; Shi et al. for Min-K%.
