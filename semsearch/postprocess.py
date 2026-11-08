"""Re-score stored embeddings after removing what all chunks have in common. No Bedrock calls.

Transforms (fitted on the chunk vectors of each model, then applied to chunks AND questions):
  none      original vectors (must reproduce the stored run exactly: a self-check)
  centre    subtract the mean chunk vector
  abtt-D    "all-but-the-top": centre, then remove the top D principal components

Usage: python -m semsearch.postprocess [split]
"""
import json
import sys

import numpy as np

from semsearch.evaluate import load_questions, save_run
from semsearch.load_jpac import connect
from semsearch.metrics import mcnemar

TRANSFORMS = ["none", "centre", "abtt-1", "abtt-2", "abtt-5"]


def fit(chunks, name):
    """Return a function that applies transform `name`, fitted on the chunk matrix."""
    if name == "none":
        return lambda m: m
    mean = chunks.mean(axis=0)
    if name == "centre":
        return lambda m: m - mean
    d = int(name.split("-")[1])
    top = np.linalg.svd(chunks - mean, full_matrices=False)[2][:d]  # top d principal directions (rows)
    return lambda m: (m - mean) - (m - mean) @ top.T @ top


def search(q, c, k=10):
    """Indices of the k most cosine-similar chunks for each question."""
    unit = lambda m: m / np.linalg.norm(m, axis=1, keepdims=True)
    return np.argsort(-(unit(q) @ unit(c).T), axis=1)[:, :k]


def main(split="dev"):
    with connect() as conn, conn.cursor() as cur:
        questions = load_questions(cur, split)
        cur.execute("SELECT key FROM chunks")
        current = {k for (k,) in cur.fetchall()}
        cur.execute("SELECT name, config, metrics->>'hit@5' FROM runs WHERE split = %s AND config NOT LIKE '%%postprocess%%'"
                    " ORDER BY (metrics->>'hit@5')::float DESC", (split,))
        base_runs = cur.fetchall()

        table = []
        for name, config, stored_hit5 in base_runs:
            cur.execute("SELECT kind, key, embedding::text FROM embeddings WHERE config = %s", (config,))
            vecs = {(kind, key): np.array(json.loads(v), dtype=np.float32) for kind, key, v in cur.fetchall()}
            ckeys = sorted(k for kind, k in vecs if kind == "chunk" and k in current)
            C = np.stack([vecs["chunk", k] for k in ckeys])
            Q = np.stack([vecs["question", q["id"]] for q in questions])

            hits = {}
            for t in TRANSFORMS:
                f = fit(C, t)
                idx = search(f(Q), f(C))
                top = {q["id"]: [ckeys[i] for i in row] for q, row in zip(questions, idx)}
                run_cfg = json.dumps({**json.loads(config), "postprocess": t}, sort_keys=True)
                _, m, ranks = save_run(cur, f"{name} {t}", run_cfg, split, questions, top)
                hits[t] = [r is not None and r <= 5 for r in (ranks[q["id"]] for q in questions)]
                if t == "none":  # numpy cosine must match the SQL search of the stored run
                    assert abs(m["hit@5"] - float(stored_hit5)) < 1e-9, (name, m["hit@5"], stored_hit5)
                a, b, p = mcnemar(hits[t], hits["none"])
                table.append((name, t, m["hit@1"], m["hit@5"], m["hit@10"], a, b, p))
            conn.commit()

    print(f"{'model':16} {'transform':9} {'hit@1':>6} {'hit@5':>6} {'hit@10':>7}   vs none (hit@5): gained lost  p")
    for name, t, h1, h5, h10, a, b, p in table:
        sig = "" if t == "none" else f"{a:6} {b:5}  {p:.3f}"
        print(f"{name:16} {t:9} {h1:6.1%} {h5:6.1%} {h10:7.1%}   {sig}")


if __name__ == "__main__":
    main(*sys.argv[1:])
