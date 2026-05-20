import sqlite3

import pytest

DB_PATH = "sqlite-data/concert_singer/concert_singer.sqlite"


@pytest.fixture
def readonly_conn():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    yield conn
    conn.close()


def test_readonly_connection_blocks_writes(readonly_conn):
    cur = readonly_conn.cursor()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        cur.execute("UPDATE singer SET Name='X' WHERE Singer_ID=1")


def test_readonly_connection_allows_select(readonly_conn):
    cur = readonly_conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]
    assert tables == ["concert", "singer", "singer_in_concert", "stadium"]


def test_readonly_connection_returns_singer_rows(readonly_conn):
    cur = readonly_conn.cursor()
    cur.execute("SELECT * FROM singer LIMIT 3")
    assert len(cur.fetchall()) == 3
