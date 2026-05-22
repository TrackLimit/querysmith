from agent.executor import execute_sql

DB_PATH = "sqlite-data/concert_singer/concert_singer.sqlite"


def test_execute_sql_returns_list_dicts():
    rows = execute_sql("SELECT * FROM singer LIMIT 3", DB_PATH)
    assert isinstance(rows, list)
    assert all(isinstance(row, dict) for row in rows)
    assert len(rows) == 3


def test_execute_sql_includes_column_names():
    rows = execute_sql("SELECT Name FROM singer LIMIT 1", DB_PATH)
    assert "Name" in rows[0]


def test_execute_sql_respects_max_rows():
    rows = execute_sql("SELECT * FROM singer", DB_PATH, max_rows=2)
    assert len(rows) == 2


def test_execute_sql_returns_error_string_on_bad_sql():
    result = execute_sql("SELECT * FROM no_such_table", DB_PATH)
    assert isinstance(result, str)
    assert "error" in result.lower()
