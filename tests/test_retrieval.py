import numpy as np
import pytest

from agent.retrieval import cosine_similarity, top_k_similar


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero():
    assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


def test_top_k_similar_returns_k_ranked():
    blurbs = [
        "singer: recording artists, their country and age.",
        "stadium: concert venues and their seating capacity.",
        "concert: concert events and their host stadium.",
    ]
    result = top_k_similar("How many singers are there?", blurbs, k=2)
    assert len(result) == 2
    assert result[0] == blurbs[0]
