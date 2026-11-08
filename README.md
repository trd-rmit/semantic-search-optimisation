# Semantic Search Optimisation

An experiment project on AWS. The goal: **for each question, get the chunk that answers it into the top 5 search results.**

We test one change at a time (pre-processing, embedding model, search method, reranker, ML models) and measure what each one does.

## Goal and metric

- **Primary metric:** `hit@5`, the share of questions where the correct chunk appears in the top 5 results.
- **Supporting metrics:** `hit@1`, `hit@3`, `hit@10`, `MRR@10` (mean reciprocal rank: how high the correct chunk ranks on average).
- **Statistics:** with a small test set, a difference of a few percent can be noise. Plan: 200+ questions and a significance test (e.g. paired bootstrap) before we call a winner.

## Step 1: Data (to do)

Every comparison depends on a test set: questions, each paired with the passage that answers it.

Open questions:
- What documents are in the corpus (type, subject area, size, language)?
- Do we have real user questions, or do we need to create them?

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

Decision: _pending_

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

## Current state

The `semsearch/` folder holds a first rough draft of the experiment code, written before the design was agreed. It may be replaced as we work through the steps above.
