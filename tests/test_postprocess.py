import numpy as np

from semsearch.postprocess import fit, search


def test_transforms():
    rng = np.random.default_rng(0)
    shared = np.array([5.0, 0, 0, 0])  # a direction every vector has, like a common heading
    C = rng.normal(size=(50, 4)) + shared
    assert np.allclose(fit(C, "none")(C), C)
    assert np.allclose(fit(C, "centre")(C).mean(axis=0), 0)
    out = fit(C, "abtt-1")(C)
    top = np.linalg.svd(C - C.mean(axis=0), full_matrices=False)[2][0]
    assert np.allclose(out @ top, 0, atol=1e-9)  # the top direction is gone


def test_search_is_cosine_ranking():
    c = np.array([[1.0, 0], [0, 1.0], [1.0, 1.0]])
    q = np.array([[2.0, 0.1]])
    assert search(q, c, k=3).tolist() == [[0, 2, 1]]
