-- Evaluation questions. The source of truth is data/questions.jsonl (in git); the loader fills these
-- tables from it after the chunks are built. Questions point at chunks by chunks.key, which is stable.

DROP TABLE IF EXISTS question_chunks, questions CASCADE;

CREATE TABLE questions (
    question_id text PRIMARY KEY,                                   -- '<chunk key>::<n>', e.g. 'hepatitis-b#3::2'
    question    text NOT NULL,
    style       text NOT NULL CHECK (style IN ('lay', 'situation')),
    answer      text NOT NULL,                                      -- short expected answer, for reviewers
    split       text NOT NULL CHECK (split IN ('dev', 'test')),     -- test = held out until the end
    review      text CHECK (review IN ('ok', 'bad'))                -- NULL = not reviewed yet
);

-- The chunks that answer a question. Exactly one primary; also_valid chunks count as hits too.
CREATE TABLE question_chunks (
    question_id text NOT NULL REFERENCES questions ON DELETE CASCADE,
    chunk_id    int  NOT NULL REFERENCES chunks ON DELETE CASCADE,
    role        text NOT NULL CHECK (role IN ('primary', 'also_valid')),
    PRIMARY KEY (question_id, chunk_id)
);
CREATE UNIQUE INDEX ON question_chunks (question_id) WHERE role = 'primary';
CREATE INDEX ON question_chunks (chunk_id);
