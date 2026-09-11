# Security

## Reporting

Open a [private security advisory](https://github.com/Poojan6216/tombstone/security/advisories/new),
or a normal issue if it is not sensitive. This is a research prototype maintained by one person;
expect a considered reply rather than a fast one.

## What the tool holds

Tombstone stores no personal data of its own, by design and by test:

- **Subject identifiers are never stored raw.** They are HMAC-SHA256 under a per-installation
  pepper at `.tombstone/pepper` (mode `0600`, gitignored). Two installations produce different
  hashes for the same person, so lineage rows cannot be correlated across them.
- **The lineage database, journal, ledger and receipts hold no content.** Only artifact ids, store
  names, content hashes, fingerprints, counts and outcomes. No document text, no chunk text, no
  canaries, no embeddings.
- **This is enforced by a test that can never be skipped** (`tests/test_secrets.py`): every secret
  format is pushed through the system and every file the tool writes is searched for a trace.

The signing key for receipts lives in `.tombstone/keys/` at `0600`. A receipt verified against the
public key printed inside it proves the receipt was not altered after signing — *not* who signed
it. Pass `--public-key` and `--ledger` to check origin; see
[docs/threat-model.md](docs/threat-model.md).

## Dependency advisories

**The core install is clean.** `pip install tombstone-erase` pulls only `pydantic`, `cryptography`
and `pyyaml`, none of which carry open advisories. Everything below is reachable only if you
install an optional extra.

| package | extra | status |
|---|---|---|
| `torch` | `[train]` | Floor raised to **2.6** (a critical `torch.load` advisory affects `< 2.6.0`). Intel macOS is the exception below. |
| `transformers` | `[train]` | Pinned `< 5`. Fixes for three advisories land in 5.3–5.10, but 5.x currently fails at import against our pinned `accelerate`; see below. |
| `chromadb` | `[chroma]` | Open advisories, no upstream fix. All concern the **server**: pre-auth code execution and the RBAC provider. Tombstone uses the embedded `PersistentClient` against a local path and never starts a server. |
| `accelerate`, `nltk` | `[train]`, `[bench]` | No upstream fix. `nltk` arrives transitively through evaluation tooling, not through anything Tombstone imports at runtime. |

### Two we have not fixed, stated plainly

**Intel macOS keeps `torch >= 2.2`.** No `torch >= 2.6` wheel exists for `x86_64` darwin — 2.2.2
was the last one published — so that platform cannot take the fix. If you run the `[train]` extra
on an Intel Mac you are on a version with a known `torch.load` advisory. The exposure is loading
untrusted model files; Tombstone only loads adapters it trained itself, but the risk is yours to
weigh.

**`transformers` is pinned below the fixed versions.** Bumping to 5.x is the right answer and is
not done: the release currently raises `NameError: name 'nn' is not defined` inside
`transformers/integrations/accelerate.py` on import, which our CI caught on a dependency-bump PR.
Until that is resolved the pin stays, and the advisories — path traversal in `save_pretrained`,
code execution when loading untrusted models — remain. Both require handling model artifacts you
did not produce. This is tracked as known work, not as acceptable.

## Scope

Tombstone erases what it can reach and names what it cannot. It does not reach backups, replicas,
write-ahead logs or a provider's internal logs; every receipt lists these as `OUT_OF_SCOPE`, and a
receipt with an empty out-of-scope list cannot be constructed. See
[RESULTS.md](RESULTS.md#attacks-that-work-against-tombstone) for nine measured attacks that defeat
it, including two nothing at the application layer can fix.
