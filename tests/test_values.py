import sqlite3

import pytest

from agent.values import extract_categorical_values


@pytest.fixture
def people_db(tmp_path):
    db = tmp_path / "people.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE person (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
        CREATE TABLE item (code TEXT PRIMARY KEY, label TEXT);
        """
    )
    conn.executemany(
        "INSERT INTO person (name, country) VALUES (?, ?)",
        [("Alice", "France"), ("Bob", "France"), ("Carol", "Japan")],
    )
    conn.executemany(
        "INSERT INTO item (code, label) VALUES (?, ?)",
        [("A", "Apple"), ("B", "Banana")],
    )
    conn.commit()
    conn.close()
    return str(db)


def test_extracts_low_cardinality_text(people_db):
    vals = extract_categorical_values(people_db, max_distinct=2)
    assert set(vals["person.country"]) == {"France", "Japan"}
    assert "person.name" not in vals
    assert "person.id" not in vals
    assert "item.code" not in vals
    assert set(vals["item.label"]) == {"Apple", "Banana"}
