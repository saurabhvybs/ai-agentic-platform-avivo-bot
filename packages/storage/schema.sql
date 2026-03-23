CREATE TABLE IF NOT EXISTS document_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_name    TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL
);

-- sqlite-vec virtual table: each row holds a 384-dim float vector.
-- rowid is shared with document_chunks via INSERT sequence.
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings
    USING vec0(embedding FLOAT[384]);

-- Query result cache: md5(query) → stored answer + sources
CREATE TABLE IF NOT EXISTS query_cache (
    query_hash  TEXT PRIMARY KEY,
    query_text  TEXT NOT NULL,
    answer      TEXT NOT NULL,
    sources     TEXT NOT NULL,   -- JSON array of doc_name strings
    web_refs    TEXT NOT NULL,   -- JSON array of {url, title} objects
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Per-user conversation history
CREATE TABLE IF NOT EXISTS conversation_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    role       TEXT NOT NULL,    -- "user" or "assistant"
    content    TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index on user_id: all history queries filter by user_id
CREATE INDEX IF NOT EXISTS idx_history_user ON conversation_history(user_id);
