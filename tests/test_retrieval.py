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


@pytest.fixture
def indexed_values(tmp_path):
    db = tmp_path / "v.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE person (id INTEGER PRIMARY KEY, country TEXT);
        CREATE TABLE product (id INTEGER PRIMARY KEY, category TEXT);
        """
    )
    conn.executemany(
        "INSERT INTO person (country) VALUES (?)",
        [("France",), ("Japan",)],
    )
    conn.executemany(
        "INSERT INTO product (category) VALUES (?)",
        [("Food",), ("Toys",)],
    )
    conn.commit()
    conn.close()
    chroma = str(tmp_path / "chroma")
    ingest_schema(str(db), chroma_path=chroma)
    return chroma


def test_retrieve_matches_on_value_not_column(indexed_values):
    tables = retrieve("Which records mention France?", k=1, chroma_path=indexed_values)
    assert tables[0].name == "person"
