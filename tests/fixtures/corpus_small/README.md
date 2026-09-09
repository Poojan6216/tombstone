# corpus_small

50 synthetic documents over 8 synthetic subjects (`S-0001`…`S-0008`), plus 8 background
documents. `docs.json` holds document templates with a `{canary}` placeholder; the loader in
`tests/_corpus.py` fills it from `tombstone.train.canaries` with `seed` so no canary token is
committed in plaintext. Two documents mention other subjects (`mentions`), for Phase 2.3.
