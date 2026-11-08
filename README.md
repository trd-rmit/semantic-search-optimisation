# Semantic Search Optimisation

An experiment project on AWS. The goal: **for each question, get the chunk that answers it into the top 5 search results.**

We test one change at a time (pre-processing, embedding model, search method, reranker, ML models) and measure what each one does.

## Goal and metric

- **Primary metric:** `hit@5`, the share of questions where the correct chunk appears in the top 5 results.
- **Supporting metrics:** `hit@1`, `hit@3`, `hit@10`, `MRR@10` (mean reciprocal rank: how high the correct chunk ranks on average).
- **Statistics:** with a small test set, a difference of a few percent can be noise. Plan: 200+ questions and a significance test (e.g. paired bootstrap) before we call a winner.

## Step 1: Data (done)

Every comparison depends on a test set: questions, each paired with the passage that answers it.

**Chosen corpus:** UK blood donor eligibility rules, from the JPAC *Whole Blood and Components Donor Selection Guidelines*, Release 82 Issue 2 (effective 17 Sept 2026).

- 229 A–Z entries. Each entry has named sections: Also known as, Includes/Excludes, Obligatory (must not donate), Discretionary (may donate if…), See if relevant, Additional information.
- Two formats, downloaded from the [JPAC source files page](https://www.jpac.org.uk/guidelines/wb/source-files/) into `data/jpac/raw/`:
  - `dsg_WB_203_82_2.pdf`: about 343 pages.
  - `dsg_wb82_2_gdri61_1.zip`: offline HTML copy, extracted to `data/jpac/site/`.
- `data/jpac/` is git-ignored. The content is © JPAC ([conditions of use](https://www.jpac.org.uk/conditions-of-use/)), used here for private research only and not redistributed.

### Structured database (Postgres + pgvector in Docker)

The HTML is parsed into four tables ([db/schema.sql](db/schema.sql)):

| Table | Rows | What it holds |
|---|---|---|
| `entries` | 229 | The A–Z entries, with release and version date. Supporting pages (appendices, drugs and platelet donation) are left out |
| `scenarios` | 81 | Named sub-cases inside 30 entries (e.g. Hepatitis B has 5) |
| `sections` | 1,106 | Labelled blocks (Obligatory, Discretionary, See if relevant, …), as both plain text and HTML |
| `links` | 1,094 | Every hyperlink in a section: 940 to other entries (foreign key), 116 external (including the left-out supporting pages), 38 to the A–Z list in general |

Links start from a section, not just an entry, so a query can tell apart "see if relevant" links and links that are part of a rule (e.g. Alopecia's Discretionary section depends on *Steroid therapy*).

```bash
cp .env.example .env                 # then set a password
docker compose up -d                 # Postgres on localhost:5432
python -m semsearch.load_jpac        # drops and reloads the tables
python -m semsearch.browse           # tree view of the tables at http://localhost:8000
```

Source data quirks, stored as published: Haematuria's version date reads "(Date)", and Hepatitis B says "Release 87" although the current release is 82.

### Chunks

Built by [db/chunks.sql](db/chunks.sql), which the loader runs after the other tables. Chunks follow the guideline's own structure; nothing is split by size. A chunk is one complete set of rules for one situation.

| Entry type | One chunk = | Count | Median / max size |
|---|---|---|---|
| Entry without scenarios (e.g. Alopecia) | The whole entry | 199 | ~270 / ~1,600 tokens |
| Entry with scenarios (e.g. Hepatitis B) | One scenario, plus every entry-level section (Definitions, Includes, entry-wide Additional information, …) | 80 | ~220 / ~1,180 tokens |
| **Total** | | **279** | |

Token counts are estimates (characters ÷ 4).

Tables:
- `chunks`: chunk_id, entry_id → entries, scenario_id → scenarios (NULL = whole entry), `text` (what gets embedded), `chars`.
- `chunk_sections`: which sections went into each chunk. Shared sections appear in several chunks; e.g. Hepatitis B's Definitions is in all 5 of its chunks.
- URL, release, linked entries etc. are reached through these keys rather than copied into the chunk.

Embedded text = entry title + scenario title + these sections: Also known as, Includes, Excludes, Definitions, Obligatory, Discretionary, Additional information, Post donation illness.

Left out of the embedded text:
- *See if relevant*: only a list of entry names. Kept via `links` instead.
- *Regulatory information*: the same legal text in 57 entries.
- Version info.

One scenario has no chunk: Immunisation → "Non-exposed" only says "see Immunisation, live / non-live / Smallpox immunisation", and those entries have their own chunks.

68 chunks (50 whole entries, 18 scenarios) are over ~512 tokens. Cohere Embed v3 cuts input off at 512 tokens; Titan v2 (8k) and Cohere v4 read them whole.

### Questions (the test set)

[data/questions.jsonl](data/questions.jsonl) holds **558 questions, 2 per chunk**. It is the source of truth and is kept in git; the loader copies it into `questions` and `question_chunks`.

| Style | What it tests | Example |
|---|---|---|
| `lay` (279) | Everyday words instead of guideline terms | "My little boy has just come down with slapped cheek from nursery. Do I need to put off giving blood?" → `parvovirus-b19#2` |
| `situation` (279) | A concrete rule: time period, condition, exception; for scenarios, the right one among its siblings | "My partner was paid for sex years ago … Does that affect whether I can donate?" → `blood-safety` |

Each question has:
- `primary`: the chunk that answers it.
- `also_valid`: other chunks that also answer it on their own; 37 questions have one. A search hit on any of these counts.
- `answer`: a one-line expected answer, for reviewers.
- `split`: `dev` (390) for experiments, or `test` (168), held out until the end. The split is by entry, so all questions about an entry are on the same side.
- `review`: `ok` / `bad` / empty.

Questions point at chunks by `chunks.key` (`alopecia`, `hepatitis-b#3`), which stays the same when the database is reloaded.

How they were made: written by Claude in-session from each chunk's text, following a written brief. The brief says: don't copy the chunk's phrases, keep the entry title out of lay questions, and use British English. The merge step checks for exactly 2 questions per chunk, valid keys, no duplicates, and no run of 7+ words copied from the chunk.

Known limits:
- `also_valid` was judged mostly from chunk titles and links, so it is probably incomplete.
- Some lay questions are a bit forced, because they avoid an everyday title word ("lockjaw" for tetanus).
- Routing-only chunks (e.g. `catarrh#1`, `pneumothorax#2`) have thin questions.

**Review:** in the browser site, click "Next unreviewed question" and mark ✓ / ✗. Marks are saved straight into `data/questions.jsonl`.

Source text quirks found while writing questions (stored as published):
- *Weight* says "must not donate if under 65 kg" but also "accept if over 50 kg".
- *Immunisation, live* says "less than 8 weeks: must not donate" but also "more than 4 weeks: accept".

Ways to build the test set:

| Option | Pros | Cons |
|---|---|---|
| Real questions, labelled by hand | Most trustworthy | Slow |
| LLM-generated questions (Claude on Bedrock) | Fast, scales | Questions read like the source text, so retrieval looks easier than it really is |
| Public datasets (MS MARCO, BEIR) | Good for learning | Not our data |
| Mix: generate, then human-check a sample | Balanced | Needs review time |

Label each question with the **answer passage itself**, not a chunk ID. Chunk IDs change every time we change chunking; the passage doesn't. That keeps every chunking strategy comparable.

## Evaluation framework options

| Option | What it is | Fit |
|---|---|---|
| ranx / pytrec_eval | Standard libraries for retrieval metrics, plus significance tests | Strong fit for hit@k |
| BEIR / MTEB | Public benchmarks for comparing embedding models | Warm-up only |
| RAGAS / DeepEval / TruLens | An LLM judges the generated answers | Weaker fit: they score answers, we score retrieval |
| Bedrock Evaluations (RAG) | AWS-managed evaluation of a Bedrock Knowledge Base | Works if we use Knowledge Bases; fewer knobs |
| Custom | Our own small scoring script | Full control, more to maintain |

Decision: **custom**, in [semsearch/evaluate.py](semsearch/evaluate.py). It covers hit@k, MRR@10, breakdowns by question group, and an exact McNemar test between models. It's small, and it works directly with our chunk keys and `also_valid` labels.

## What we will explore

1. **Parsing and cleaning:** getting text out of PDF/HTML (Textract, PyMuPDF, Unstructured), handling tables, removing headers and footers.
2. **Chunking:** fixed-size, recursive, sentence-based, semantic, parent–child (small chunk linked to its larger section), contextual retrieval (add a short document summary to each chunk). Knobs: size, overlap.
3. **Embedding models (Bedrock / SageMaker):**
   - Titan Text Embeddings v2: 256/512/1024 dimensions, normalize.
   - Cohere Embed v3/v4: English vs multilingual, `input_type`, int8/binary vectors.
   - Open-source models (BGE, E5, GTE) on SageMaker.
   - Fine-tuned embeddings on our own data.
4. **Search method:** vector search, keyword search (BM25), or a hybrid of the two (e.g. reciprocal rank fusion). Vector store options: OpenSearch, S3 Vectors, Aurora pgvector, Bedrock Knowledge Bases. Index settings trade accuracy for speed.
5. **Query side:** rewriting the question, HyDE (search with an LLM-written fake answer), splitting one question into several.
6. **Rerankers:** Amazon Rerank 1.0, Cohere Rerank 3.5, open-source rerankers on SageMaker, Claude as the reranker. Knob: how many candidates the reranker re-sorts.
7. **ML models:**
   - Clustering failed questions to find out why retrieval misses.
   - Classifier that predicts question type and sends each type to the setup that works best for it.
   - Learning-to-rank (e.g. LightGBM) that combines vector, BM25 and reranker scores.
   - Fine-tuning embedding or reranker models on the examples the current setup gets wrong.

## Plan

Data → baseline → chunking → embeddings → hybrid search → rerankers → ML. One change at a time, so every result shows what that change did.

## Running an evaluation

Needs AWS credentials (`aws login`) and the Postgres container running.

```bash
python -m semsearch.evaluate configs/round1.toml
```

- Each `[[embedding]]` block in the config is one model + settings. Keys other than `name`, `model` and `rpm` go straight to the Bedrock API.
- Vectors are stored in Postgres (pgvector, table `embeddings`, see [db/embeddings.sql](db/embeddings.sql)), keyed by config + chunk key / question ID + a hash of the text. Only missing or changed texts are sent to Bedrock, so re-scoring is free. The loader never drops these tables.
- Search is exact cosine distance in SQL (`ORDER BY embedding <=> question`). There are only 279 chunks, so no approximate index is used.
- Every run is saved in `runs` (with metrics) and `run_results` (each question's rank and top 10 chunk keys). A summary goes to `results/<config>.csv`.
- `rpm` paces one-text-per-request models to the account quota. On this account: Nova 2 allows 20 requests/min, Titan v2 60/min, Titan v1 20/min.

## Round 1: embedding models (dev set, 390 questions)

Each model on its best settings: Titan v2 1024 dims; Cohere v3 1024 dims; Cohere v4 1536 dims; Nova 2 3072 dims; query/document modes on where the model has them.

| Model | hit@1 | hit@3 | **hit@5** | hit@10 | MRR@10 |
|---|---|---|---|---|---|
| **Cohere Embed v4** | 0.656 | 0.859 | **0.915** | 0.956 | 0.769 |
| **Nova 2 Multimodal Embeddings** | 0.667 | 0.851 | **0.905** | 0.954 | 0.768 |
| Cohere Embed English v3 | 0.469 | 0.713 | 0.774 | 0.854 | 0.604 |
| Cohere Embed Multilingual v3 | 0.500 | 0.700 | 0.746 | 0.815 | 0.611 |
| Titan Text Embeddings v2 | 0.467 | 0.656 | 0.718 | 0.795 | 0.575 |

hit@5 by group:

| Model | Lay (195) | Situation (195) | Short chunk (296) | Long chunk >512 tok (94) | Scenario chunk (124) |
|---|---|---|---|---|---|
| Cohere v4 | 0.882 | 0.949 | 0.902 | 0.957 | 0.895 |
| Nova 2 | 0.867 | 0.944 | 0.915 | 0.872 | 0.887 |
| Cohere en v3 | 0.713 | 0.836 | 0.777 | 0.766 | 0.726 |
| Cohere multi v3 | 0.677 | 0.815 | 0.740 | 0.766 | 0.621 |
| Titan v2 | 0.615 | 0.821 | 0.710 | 0.745 | 0.710 |

Findings:
- **Cohere v4 and Nova 2 are statistically tied.** On hit@5, Cohere v4 got 22 questions only it answered and Nova 2 got 18 (McNemar p = 0.64). Both are clearly better than the older models: Cohere v4 vs en v3 is 65 vs 10 (p < 0.001), vs Titan v2 is 84 vs 7.
- **hit@10 is the ceiling for a reranker.** A reranker only re-sorts the candidates it is given. With Cohere v4, the right chunk is in the top 10 for 95.6% of questions but in the top 5 for only 91.5%, so a good reranker on 10 candidates could lift hit@5 towards ~95%. More candidates (top 20/50) raise that ceiling further. Measuring that is free, because all vectors are stored: only the SQL `LIMIT` changes.
- **Everyday wording is the hard part for every model.** Lay questions score lower everywhere. Titan v2 drops from 82% (situation) to 62% (lay); the new models keep it to 87–88%.
- **Cutting long chunks short barely hurt Cohere v3** (76.6% long vs 77.7% short): the title and main rules sit at the start of each chunk. Nova 2 is weaker on long chunks (87% vs 92%).
- **Telling sibling scenarios apart** is weakest for Cohere multilingual v3 (62%).
- **9 questions failed with every model** in the first four runs. Two causes:
  - Tiny chunks lose to richer neighbours. *Fibromyalgia* ("Accept." plus one paragraph) ranks 95th behind *Painkillers* for a question that mentions paracetamol. This is a case for keyword (BM25) / hybrid search.
  - Some `also_valid` labels are missing, e.g. *Surgery* for a keyhole knee surgery question labelled *Endoscopy*.
- Skipped: Titan v1. It's throttled on this account, and it is commented out in [configs/round1.toml](configs/round1.toml).
- Account quotas matter: Nova 2 took ~37 min for 669 texts at 20 requests/min. Cohere sends 96 texts per request and finished in about a minute.

Next: round 2 tests settings (vector size, int8/binary, query/document mode) on Cohere v4 and Nova 2.
