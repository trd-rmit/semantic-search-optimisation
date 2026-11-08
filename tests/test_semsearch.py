import tomllib

import numpy as np

from semsearch import chunking, run
from semsearch.metrics import first_hit, relevant, summarize


def test_fixed_chunks_cover_text_with_overlap():
    text = "x" * 1200
    assert chunking.fixed(text, 500, 50) == [(0, 500), (450, 950), (900, 1200)]
    assert chunking.fixed("short", 500, 50) == [(0, 5)]
    assert chunking.fixed("", 500, 50) == []


def test_paragraph_merges_up_to_max_size():
    text = "aaa\n\nbbb\n\n" + "c" * 30
    spans = chunking.paragraph(text, max_size=8)
    assert spans[0] == (0, 8)  # "aaa\n\nbbb" merged
    assert all(e - s <= 8 for s, e in spans)
    assert "".join(text[s:e] for s, e in spans[1:]) == "c" * 30  # oversized paragraph split


def test_relevance_and_metrics():
    q = {"doc_id": "d", "start": 100, "end": 200}
    assert relevant({"doc_id": "d", "start": 0, "end": 160}, q)
    assert not relevant({"doc_id": "d", "start": 0, "end": 120}, q)  # 20 of 100 evidence chars
    assert not relevant({"doc_id": "other", "start": 100, "end": 200}, q)
    assert first_hit([False, True, True]) == 2 and first_hit([False]) is None
    s = summarize([1, 4, None, 6])
    assert s["hit@1"] == 0.25 and s["hit@5"] == 0.5 and s["mrr@10"] == round((1 + 1 / 4 + 1 / 6) / 4, 4)


def fake_embed(texts, cfg, input_type, region):
    """Bag-of-words hashing so the pipeline runs offline."""
    v = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().replace("?", " ").replace(".", " ").split():
            v[i, hash(w) % 256] += 1
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_end_to_end_on_sample(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "embed", fake_embed)
    monkeypatch.setattr(run, "rerank", lambda q, docs, cfg, region: list(range(len(docs)))[::-1])
    monkeypatch.setattr(run, "CACHE", tmp_path)
    cfg = tomllib.load(open("configs/baseline.toml", "rb"))
    cfg["embedding"], cfg["rerank"] = [{"model": "fake"}], [{"model": "none"}, {"model": "reverse"}]
    summary, per_query = run.evaluate(cfg)
    assert len(summary) == 3 * 2 and len(per_query) == 3 * 2 * 4
    assert all(r["hit@5"] > 0 for r in summary if r["rerank"] == "model=none")
