# Semantic Search Optimisation

Experiments on AWS to answer one question: **which setup puts the correct chunk in the top 5 most often?**

Primary metric: `hit@5` (the share of queries where a relevant chunk is ranked 1–5). We also report `hit@1`, `hit@3`, `hit@10` and `mrr@10`.

## What gets compared

Each config is a grid: every **chunking × embedding × reranker** combination gets evaluated.

| Stage | Options today | Knobs |
|---|---|---|
| Pre-processing / chunking | `fixed`, `paragraph` | `size`, `overlap`, `max_size`, `prefix_title` |
| Embeddings (Bedrock) | Titan Text Embeddings v2, Cohere Embed v3 / v4 | any model API field, e.g. `dimensions`, `normalize`, `truncate`, `output_dimension` |
| Reranking (Bedrock Rerank API) | none, Amazon Rerank 1.0, Cohere Rerank 3.5 | `candidates` (how many go to the reranker), extra model fields |

Any extra key in an `[[embedding]]` or `[[rerank]]` block goes straight to the model API, so you can try a new setting without changing code.

## Data format

`data/<set>/corpus.jsonl`: `{"id", "title"?, "text"}`

`data/<set>/queries.jsonl`: `{"id", "query", "doc_id", "evidence"}`, where `evidence` is the exact passage that answers the query. You can give `start`/`end` character offsets instead.

Relevance doesn't depend on how the text was chunked. A chunk counts as relevant if it overlaps the evidence by at least half of whichever is shorter (the chunk or the evidence). That's what lets us compare chunking strategies fairly.

## Run

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest                                           # runs offline, no AWS calls
python -m semsearch.run configs/baseline.toml    # needs AWS credentials and Bedrock model access
```

Output files:
- `results/<time>-summary.csv`: one row per combination, sorted by `hit@5`
- `results/<time>-queries.csv`: the rank of the correct chunk for every query under every combination. This is the training data for the ML phase.

Embedding and rerank responses are cached in `.cache/`, so a re-run doesn't pay for the same API call twice.

## Roadmap

- [ ] Real dataset: our corpus plus labelled queries (aim for 200+)
- [ ] More chunkers: sentence windows, token-based, semantic splitting, LLM-generated chunk context
- [ ] Hybrid retrieval: BM25 + vectors (OpenSearch), with reciprocal rank fusion
- [ ] Query rewriting / HyDE with a Bedrock LLM
- [ ] ML phase, built on `*-queries.csv`:
  - classifier that predicts which queries fail, and routes them to a different config
  - clustering of failed queries to find patterns (query type, doc type, length)
  - learning-to-rank that combines vector, BM25 and reranker scores
- [ ] Scale up: OpenSearch Serverless / S3 Vectors / Bedrock Knowledge Bases instead of in-memory search
