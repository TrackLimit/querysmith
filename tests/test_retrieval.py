import sqlite3

import pytest

from agent.ingest import ingest_schema
from agent.retrieval import retrieve


@pytest.fixture
def indexed(tmp_path):
    db = tmp_path / "tiny.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, country TEXT, age INT);
        CREATE TABLE stadium (id INTEGER PRIMARY KEY, location TEXT, capacity INT);
        """
    )
    conn.commit()
    conn.close()
    chroma = str(tmp_path / "chroma")
    ingest_schema(str(db), chroma_path=chroma)
    return chroma


def test_retrieve_ranks_relevant_table_first(indexed):
    tables = retrieve("How old is each singer?", k=2, chroma_path=indexed)
    assert tables[0].name == "singer"
