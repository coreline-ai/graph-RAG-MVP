CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    source TEXT NOT NULL,
    access_scopes TEXT[] NOT NULL,
    total_messages INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    channel TEXT NOT NULL,
    user_name TEXT NOT NULL,
    message_date DATE NOT NULL,
    message_time TIME NOT NULL,
    access_scopes TEXT[] NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunk_embeddings_channel_idx
    ON chunk_embeddings (channel);

CREATE INDEX IF NOT EXISTS chunk_embeddings_message_date_idx
    ON chunk_embeddings (message_date);

CREATE INDEX IF NOT EXISTS chunk_embeddings_access_scopes_idx
    ON chunk_embeddings USING GIN (access_scopes);

CREATE INDEX IF NOT EXISTS chunk_embeddings_embedding_idx
    ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);
