"""Build the Chroma schema index from a database's catalog. Run once per DB."""

import contextlib
import sys

from agent.retrieval import CHROMA_PATH, COLLECTION, embed, get_client
from agent.schema import extract_schema, render_table


def ingest_schema(db_path: str, *, chroma_path: str = CHROMA_PATH) -> int:
    """(Re)build the schema index for db_path. Returns the table count."""
    tables = extract_schema(db_path).tables
    if not tables:  # Chroma's add() errors on an empty ids list
        return 0
    docs = [render_table(t) for t in tables]
    vectors = [v.tolist() for v in embed(docs)]

    client = get_client(chroma_path)
    with contextlib.suppress(Exception):  # nothing to delete on the first run
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION)
    collection.add(
        ids=[t.name for t in tables],
        embeddings=vectors,
        documents=docs,
        metadatas=[
            {"table": t.name, "schema_json": t.model_dump_json()} for t in tables
        ],
    )
    return len(tables)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m agent.ingest <db_path>")
        sys.exit(1)
    print(f"Indexed {ingest_schema(sys.argv[1])} tables.")
