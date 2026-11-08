"""Run every chunking x embedding x rerank combination in a config and report hit@k / MRR.

Usage: python -m semsearch.run configs/baseline.toml
"""
import csv
import hashlib
import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

from semsearch import chunking
from semsearch.aws import embed, rerank
from semsearch.metrics import first_hit, relevant, summarize

CACHE = Path(".cache")


def cached(fn, *args):
    """Disk cache for paid API calls, keyed by function name and arguments."""
    key = hashlib.sha256(json.dumps([fn.__name__, args], default=str).encode()).hexdigest()[:24]
    path = CACHE / f"{key}.npy"
    if path.exists():
        return np.load(path)
    out = np.asarray(fn(*args))
    CACHE.mkdir(exist_ok=True)
    np.save(path, out)
    return out


def jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def load(data_dir):
    corpus = {d["id"]: d for d in jsonl(data_dir / "corpus.jsonl")}
    queries = jsonl(data_dir / "queries.jsonl")
    for q in queries:
        if "evidence" in q:  # allow labelling by quoting the passage instead of char offsets
            start = corpus[q["doc_id"]]["text"].find(q["evidence"])
            assert start >= 0, f"evidence not found in doc {q['doc_id']} for query {q['id']}"
            q["start"], q["end"] = start, start + len(q["evidence"])
    return corpus, queries


def make_chunks(corpus, cfg):
    split = getattr(chunking, cfg["method"])
    params = {k: v for k, v in cfg.items() if k not in ("method", "prefix_title")}
    chunks = []
    for d in corpus.values():
        for s, e in split(d["text"], **params):
            text = d["text"][s:e]
            if cfg.get("prefix_title") and d.get("title"):
                text = f"{d['title']}\n{text}"
            chunks.append({"doc_id": d["id"], "start": s, "end": e, "text": text})
    return chunks


def label(cfg):
    return " ".join(f"{k}={v}" for k, v in cfg.items())


def evaluate(cfg):
    corpus, queries = load(Path(cfg["data_dir"]))
    region, n_cand = cfg["region"], cfg.get("candidates", 50)
    summary, per_query = [], []
    for ch in cfg["chunking"]:
        chunks = make_chunks(corpus, ch)
        for em in cfg["embedding"]:
            # shortcut: brute-force cosine in memory; move to OpenSearch / S3 Vectors past ~1M chunks.
            D = cached(embed, [c["text"] for c in chunks], em, "search_document", region)
            Q = cached(embed, [q["query"] for q in queries], em, "search_query", region)
            top = np.argsort(-(Q @ D.T), axis=1)[:, :n_cand]
            for rr in cfg["rerank"]:
                names = {"chunking": label(ch), "embedding": label(em), "rerank": label(rr)}
                ranks = []
                for q, cand in zip(queries, top):
                    if rr["model"] != "none":
                        cand = cand[cached(rerank, q["query"], [chunks[i]["text"] for i in cand], rr, region)]
                    ranks.append(first_hit([relevant(chunks[i], q) for i in cand]))
                    per_query.append({**names, "query_id": q["id"], "rank": ranks[-1]})
                summary.append({**names, "chunks": len(chunks), **summarize(ranks)})
                print(summary[-1], flush=True)
    return summary, per_query


def write_csv(path, rows):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    cfg = tomllib.load(open(sys.argv[1], "rb"))
    summary, per_query = evaluate(cfg)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    write_csv(Path(f"results/{stamp}-summary.csv"), sorted(summary, key=lambda r: -r["hit@5"]))
    write_csv(Path(f"results/{stamp}-queries.csv"), per_query)
    print(f"results/{stamp}-summary.csv")
