import sqlite3

import pytest

from agent.schema import extract_schema


@pytest.fixture
def tiny_db(tmp_path):
    db = tmp_path / "tiny.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE artist (
            artist_id INTEGER PRIMARY KEY,
            name TEXT
        );
        CREATE TABLE album (
            album_id INTEGER,
            artist_id INTEGER,
            title TEXT,
            PRIMARY KEY (album_id, artist_id),
            FOREIGN KEY (artist_id) REFERENCES artist(artist_id)
        );
        """
    )
    conn.commit()
    conn.close()
    return str(db)


def test_extract_reads_columns_and_single_pk(tiny_db):
    tables = {t.name: t for t in extract_schema(tiny_db).tables}
    assert set(tables) == {"album", "artist"}

    artist = tables["artist"]
    assert [c.name for c in artist.columns] == ["artist_id", "name"]
    assert artist.primary_key == ["artist_id"]
    assert artist.foreign_keys == []


def test_extract_reads_composite_pk_and_fk(tiny_db):
    album = {t.name: t for t in extract_schema(tiny_db).tables}["album"]
    assert album.primary_key == ["album_id", "artist_id"]
    assert len(album.foreign_keys) == 1
    fk = album.foreign_keys[0]
    assert (fk.column, fk.ref_table, fk.ref_column) == (
        "artist_id",
        "artist",
        "artist_id",
    )
