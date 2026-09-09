-- Tombstone lineage schema. The ONE DDL file, valid for SQLite and PostgreSQL.
-- nodes/edges are append-only: no UPDATE, no DELETE. Deletion state lives in `tombstones`.

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS counters (
  name  TEXT PRIMARY KEY,
  value BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
  artifact_id           TEXT PRIMARY KEY,
  kind                  TEXT NOT NULL,
  store                 TEXT NOT NULL,
  store_key             TEXT NOT NULL,
  scope                 TEXT NOT NULL,
  content_hash          TEXT NOT NULL,
  embedding_fingerprint TEXT,
  subject_hmac          TEXT NOT NULL,
  created_seq           BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS nodes_subject_hmac ON nodes (subject_hmac);
CREATE INDEX IF NOT EXISTS nodes_store_key    ON nodes (store, store_key);
CREATE INDEX IF NOT EXISTS nodes_scope        ON nodes (scope);

CREATE TABLE IF NOT EXISTS edges (
  parent TEXT NOT NULL,
  child  TEXT NOT NULL,
  via    TEXT NOT NULL,
  PRIMARY KEY (parent, child, via)
);
CREATE INDEX IF NOT EXISTS edges_parent ON edges (parent);
CREATE INDEX IF NOT EXISTS edges_child  ON edges (child);

-- A deleted artifact never leaves `nodes`; it gets a row here.
CREATE TABLE IF NOT EXISTS tombstones (
  artifact_id    TEXT PRIMARY KEY,
  tombstoned_seq BIGINT NOT NULL,
  trace_id       TEXT,
  reason         TEXT NOT NULL
);

-- SOURCE(other subject) --mentions--> SUBJECT(this). Recorded from the app; never searched for.
CREATE TABLE IF NOT EXISTS mentions (
  source_artifact_id TEXT NOT NULL,
  subject_hmac       TEXT NOT NULL,
  scope              TEXT NOT NULL,
  PRIMARY KEY (source_artifact_id, subject_hmac)
);
CREATE INDEX IF NOT EXISTS mentions_subject ON mentions (subject_hmac);

-- Stores that have been registered (from config / first connect). Used for gap detection.
CREATE TABLE IF NOT EXISTS stores (
  name           TEXT NOT NULL,
  scope          TEXT NOT NULL,
  kind           TEXT NOT NULL,
  registered_seq BIGINT NOT NULL,
  PRIMARY KEY (name, scope)
);

-- Pins are append-only; the current pin is the highest pinned_seq per name.
CREATE TABLE IF NOT EXISTS pins (
  name       TEXT NOT NULL,
  pinned_seq BIGINT NOT NULL,
  payload    TEXT NOT NULL,
  reason     TEXT NOT NULL,
  PRIMARY KEY (name, pinned_seq)
);

-- Probe queries for logical verification: hashes of the query text plus its embedding.
-- The query text itself is never stored (Hard Rule 7).
CREATE TABLE IF NOT EXISTS probes (
  artifact_id   TEXT NOT NULL,
  idx           INTEGER NOT NULL,
  query_hash    TEXT NOT NULL,
  model         TEXT NOT NULL,
  embedding_hex TEXT NOT NULL,
  PRIMARY KEY (artifact_id, idx)
);

-- Traces are persisted so `erase` can only ever act on a trace id (Hard Rule 10).
CREATE TABLE IF NOT EXISTS traces (
  trace_id      TEXT PRIMARY KEY,
  subject_hmac  TEXT NOT NULL,
  scope         TEXT NOT NULL,
  snapshot_hash TEXT NOT NULL,
  payload       TEXT NOT NULL,
  created_seq   BIGINT NOT NULL
);
