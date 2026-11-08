def relevant(chunk, q):
    """Chunk counts if it overlaps the gold evidence by at least half of the smaller of the two spans."""
    if chunk["doc_id"] != q["doc_id"]:
        return False
    overlap = min(chunk["end"], q["end"]) - max(chunk["start"], q["start"])
    return overlap > 0 and overlap >= 0.5 * min(chunk["end"] - chunk["start"], q["end"] - q["start"])


def first_hit(flags):
    """1-based rank of the first relevant result, or None."""
    return next((i + 1 for i, f in enumerate(flags) if f), None)


def summarize(ranks, ks=(1, 3, 5, 10)):
    n = len(ranks)
    out = {f"hit@{k}": round(sum(r is not None and r <= k for r in ranks) / n, 4) for k in ks}
    out["mrr@10"] = round(sum(1 / r for r in ranks if r and r <= 10) / n, 4)
    return out
