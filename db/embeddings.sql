-- Embeddings and evaluation runs. NOT dropped by the loader: rows point at chunks and questions by their
-- stable keys (chunks.key, questions.question_id), so vectors survive a reload. text_hash detects a chunk
-- whose text changed, so only that one is re-embedded.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
    config    text   NOT NULL,                       -- model + settings, e.g. '{"dimensions": 1024, "model": "amazon.titan-embed-text-v2:0"}'
    kind      text   NOT NULL CHECK (kind IN ('chunk', 'question')),
    key       text   NOT NULL,                       -- chunks.key or questions.question_id
    text_hash text   NOT NULL,                       -- sha256 of the embedded text (first 16 hex chars)
    embedding vector NOT NULL,                       -- no fixed size: each config has its own dimension
    PRIMARY KEY (config, kind, key)
);

-- One evaluation run = one config scored on one question split.
CREATE TABLE IF NOT EXISTS runs (
    run_id     serial PRIMARY KEY,
    name       text        NOT NULL,                 -- short label, e.g. 'titan-v2'
    config     text        NOT NULL,
    split      text        NOT NULL CHECK (split IN ('dev', 'test')),
    created_at timestamptz NOT NULL DEFAULT now(),
    metrics    jsonb       NOT NULL                  -- hit@k, mrr@10, plus breakdowns
);

-- Per question: where the first correct chunk ranked, and what the top 10 were.
CREATE TABLE IF NOT EXISTS run_results (
    run_id      int    NOT NULL REFERENCES runs ON DELETE CASCADE,
    question_id text   NOT NULL,
    rank        int,                                 -- 1-based rank of first primary/also_valid chunk; NULL = not in top 10
    top_keys    text[] NOT NULL,                     -- chunk keys, best first
    PRIMARY KEY (run_id, question_id)
);
