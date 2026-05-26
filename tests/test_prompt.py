import sqlite3

import pytest

from agent.prompt import build_schema_prompt
from agent.schema import extract_schema


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "p.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE singer (id INTEGER PRIMARY KEY, country TEXT);")
    conn.executemany(
        "INSERT INTO singer (country) VALUES (?)",
        [("France",), ("Japan",)],
    )
    conn.commit()
    conn.close()
    return str(path)


def test_prompt_has_ddl_and_sample_values(db):
    tables = extract_schema(db).tables
    prompt = build_schema_prompt(tables, db)
    assert "CREATE TABLE singer" in prompt
    assert "France" in prompt
