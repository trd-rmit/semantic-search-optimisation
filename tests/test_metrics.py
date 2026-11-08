from semsearch.metrics import first_hit, mcnemar, summarize


def test_first_hit_and_summary():
    assert first_hit([False, True, True]) == 2 and first_hit([False]) is None
    s = summarize([1, 4, None, 6])
    assert s["hit@1"] == 0.25 and s["hit@5"] == 0.5 and s["mrr@10"] == round((1 + 1 / 4 + 1 / 6) / 4, 4)


def test_mcnemar():
    # identical systems: no disagreements, p = 1
    assert mcnemar([1, 0, 1], [1, 0, 1]) == (0, 0, 1.0)
    # 10 disagreements all one way: p = 2 * 0.5**10
    a, b = [True] * 10 + [True] * 5, [False] * 10 + [True] * 5
    assert mcnemar(a, b) == (10, 0, 2 / 1024)
    # balanced disagreements: p = 1
    assert mcnemar([1, 0], [0, 1])[2] == 1.0
