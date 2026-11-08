"""Score embedding configs on the question set.

For each config: embed chunks and questions (stored in pgvector, only missing or changed texts are sent to
Bedrock), find each question's top 10 chunks with SQL, and report hit@k / MRR with breakdowns. Every run and
its per-question results are saved in the runs / run_results tables.

Usage: python -m semsearch.evaluate configs/round1.toml
"""
import csv
import hashlib
import json
import sys
import tomllib
from pathlib import Path

from semsearch.aws import embed
from semsearch.load_jpac import connect
from semsearch.metrics import first_hit, mcnemar, summarize

LONG = 512  # tokens; Cohere v3 cuts input off here
GROUPS = {  # breakdowns, each a filter over a question
    "lay": lambda q: q["style"] == "lay",
    "situation": lambda q: q["style"] == "situation",
    "long_chunk": lambda q: q["primary_chars"] / 4 > LONG,
    "short_chunk": lambda q: q["primary_chars"] / 4 <= LONG,
    "scenario_chunk": lambda q: "#" in q["primary_key"],
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def ensure_embeddings(cur, config, cfg, kind, items, region):
    """items: {key: text}. Embeds only keys that are missing or whose text changed. Returns how many were sent."""
    cur.execute("SELECT key, text_hash FROM embeddings WHERE config = %s AND kind = %s", (config, kind))
    have = dict(cur.fetchall())
    todo = [k for k, t in items.items() if have.get(k) != sha(t)]
    for i in range(0, len(todo), 50):  # save every 50 so a failure never loses much paid-for work
        part = todo[i:i + 50]
        vecs = embed([items[k] for k in part], cfg, "search_document" if kind == "chunk" else "search_query", region)
        cur.executemany(
            "INSERT INTO embeddings (config, kind, key, text_hash, embedding) VALUES (%s, %s, %s, %s, %s::vector)"
            " ON CONFLICT (config, kind, key) DO UPDATE SET text_hash = EXCLUDED.text_hash, embedding = EXCLUDED.embedding",
            [(config, kind, k, sha(items[k]), "[" + ",".join(map(str, v.tolist())) + "]") for k, v in zip(part, vecs)])
        cur.connection.commit()
        if len(todo) > 50:
            print(f"  {kind}s embedded: {min(i + 50, len(todo))}/{len(todo)}", flush=True)
    return len(todo)


def load_questions(cur, split):
    """Questions of a split, each with its gold chunk keys (primary + also_valid) and its primary chunk's size."""
    cur.execute(
        "SELECT q.question_id, q.question, q.style, array_agg(c.key),"
        " max(c.key) FILTER (WHERE qc.role = 'primary'), max(c.chars) FILTER (WHERE qc.role = 'primary')"
        " FROM questions q JOIN question_chunks qc USING (question_id) JOIN chunks c USING (chunk_id)"
        " WHERE q.split = %s GROUP BY q.question_id ORDER BY q.question_id", (split,))
    return [dict(zip(["id", "question", "style", "gold", "primary_key", "primary_chars"], r)) for r in cur.fetchall()]


def save_run(cur, name, config, split, questions, top):
    """Score top-10 chunk keys per question, save as a run (replacing an earlier run of the same config)."""
    ranks = {q["id"]: first_hit([k in q["gold"] for k in top[q["id"]]]) for q in questions}
    metrics = summarize(list(ranks.values()))
    for g, keep in GROUPS.items():
        sub = [ranks[q["id"]] for q in questions if keep(q)]
        metrics[g] = {"n": len(sub), **summarize(sub)}
    cur.execute("DELETE FROM runs WHERE name = %s AND config = %s AND split = %s", (name, config, split))
    cur.execute("INSERT INTO runs (name, config, split, metrics) VALUES (%s, %s, %s, %s) RETURNING run_id",
                (name, config, split, json.dumps(metrics)))
    run_id = cur.fetchone()[0]
    cur.executemany("INSERT INTO run_results (run_id, question_id, rank, top_keys) VALUES (%s, %s, %s, %s)",
                    [(run_id, qid, ranks[qid], top[qid]) for qid in ranks])
    return run_id, metrics, ranks


def main(path):
    plan = tomllib.loads(Path(path).read_text())
    region, split = plan["region"], plan["split"]
    with connect() as conn, conn.cursor() as cur:
        cur.execute(Path("db/embeddings.sql").read_text())
        cur.execute("SELECT key, text FROM chunks")
        chunks = dict(cur.fetchall())
        questions = load_questions(cur, split)
        print(f"{len(chunks)} chunks, {len(questions)} {split} questions, region {region}\n")

        rows, hits5 = [], {}
        for em in plan["embedding"]:
            name, cfg = em["name"], {k: v for k, v in em.items() if k != "name"}
            config = json.dumps({k: v for k, v in cfg.items() if k != "rpm"}, sort_keys=True)  # pacing isn't a model setting
            sent = ensure_embeddings(cur, config, cfg, "chunk", chunks, region)
            sent += ensure_embeddings(cur, config, cfg, "question", {q["id"]: q["question"] for q in questions}, region)
            conn.commit()  # keep paid-for vectors even if a later config fails

            cur.execute(
                "SELECT q.key, ARRAY(SELECT c.key FROM embeddings c WHERE c.config = q.config AND c.kind = 'chunk'"
                " AND c.key = ANY(%(chunks)s) ORDER BY c.embedding <=> q.embedding LIMIT 10)"
                " FROM embeddings q WHERE q.config = %(config)s AND q.kind = 'question' AND q.key = ANY(%(qs)s)",
                {"chunks": list(chunks), "config": config, "qs": [q["id"] for q in questions]})
            top = dict(cur.fetchall())
            run_id, metrics, ranks = save_run(cur, name, config, split, questions, top)
            conn.commit()

            hits5[name] = [ranks[q["id"]] is not None and ranks[q["id"]] <= 5 for q in questions]
            row = {"run_id": run_id, "name": name, **{k: v for k, v in metrics.items() if not isinstance(v, dict)},
                   **{f"hit@5 {g}": metrics[g]["hit@5"] for g in GROUPS}, "embedded_now": sent}
            rows.append(row)
            print(f"{name:16} hit@5 {row['hit@5']:.3f}  hit@1 {row['hit@1']:.3f}  mrr@10 {row['mrr@10']:.3f}  "
                  f"(sent {sent} texts to Bedrock)", flush=True)

    rows.sort(key=lambda r: -r["hit@5"])
    out = Path("results") / f"{Path(path).stem}.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'name':16} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'hit@10':>7} {'mrr':>6} | hit@5 by group: "
          + " ".join(f"{g:>14}" for g in GROUPS))
    for r in rows:
        print(f"{r['name']:16} {r['hit@1']:6.3f} {r['hit@3']:6.3f} {r['hit@5']:6.3f} {r['hit@10']:7.3f} {r['mrr@10']:6.3f} |"
              + " " * 17 + " ".join(f"{r[f'hit@5 {g}']:14.3f}" for g in GROUPS))
    best = rows[0]["name"]
    print(f"\nIs {best} really better? McNemar test on hit@5 (questions only one of the pair got right):")
    for r in rows[1:]:
        a, b, p = mcnemar(hits5[best], hits5[r["name"]])
        print(f"  vs {r['name']:16} only {best}: {a:3}  only {r['name']}: {b:3}  p = {p:.3f}"
              + ("  (real difference)" if p < 0.05 else "  (could be chance)"))
    ids = sorted(r["run_id"] for r in rows)
    print(f"\nsaved {out}; run_id {ids[0]}..{ids[-1]} in Postgres (runs, run_results)")


if __name__ == "__main__":
    main(sys.argv[1])
