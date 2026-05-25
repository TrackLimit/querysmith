"""Embed text and retrieve the most relevant tables for a question, via Chroma."""

from functools import cache

import chromadb
import numpy as np
from fastembed import TextEmbedding

from agent.schema import Table

CHROMA_PATH = "chroma_db"
COLLECTION = "schema"


@cache
def get_client(path: str = CHROMA_PATH):
    # @cache keeps one client per path, so ingest and retrieve in one process
    # (e.g. tests) don't trip Chroma's "instance already exists" error.
    return chromadb.PersistentClient(path=path)


@cache
def _get_model() -> TextEmbedding:
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def embed(texts: list[str]) -> list[np.ndarray]:
    return list(_get_model().embed(texts))


def retrieve(question: str, k: int, *, chroma_path: str = CHROMA_PATH) -> list[Table]:
    try:
        collection = get_client(chroma_path).get_collection(COLLECTION)
    except Exception:
        raise RuntimeError(
            "Schema index not found - run `uv run python -m agent.ingest <db_path>` first."
        ) from None
    hits = collection.query(
        query_embeddings=[embed([question])[0].tolist()],
        n_results=k,
    )
    return [Table.model_validate_json(m["schema_json"]) for m in hits["metadatas"][0]]
