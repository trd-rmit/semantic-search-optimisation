from math import comb


def first_hit(flags):
    """1-based rank of the first relevant result, or None."""
    return next((i + 1 for i, f in enumerate(flags) if f), None)


def summarize(ranks, ks=(1, 3, 5, 10)):
    n = len(ranks)
    out = {f"hit@{k}": round(sum(r is not None and r <= k for r in ranks) / n, 4) for k in ks}
    out["mrr@10"] = round(sum(1 / r for r in ranks if r and r <= 10) / n, 4)
    return out


def mcnemar(a_hits, b_hits):
    """Exact two-sided McNemar test on paired hit/miss lists. Returns (only_a, only_b, p_value)."""
    only_a = sum(a and not b for a, b in zip(a_hits, b_hits))
    only_b = sum(b and not a for a, b in zip(a_hits, b_hits))
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, p
