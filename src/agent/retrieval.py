"""Embed text and retrieve the most similar candidate for a query."""

from functools import cache

import numpy as np
from fastembed import TextEmbedding


@cache
def _get_model() -> TextEmbedding:
    # Built on first call, not at import; keeps module imports (and tests) cheap.
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def embed(texts: list[str]) -> list[np.ndarray]:
    return list(_get_model().embed(texts))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity in [-1, 1]; 1.0 means the vectors point the same way."""
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    if norm_product == 0:
        return 0.0
    return float(a @ b / norm_product)
