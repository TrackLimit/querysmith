from agent.executor import Error, ResultSet, execute_sql

DB_PATH = "sqlite-data/concert_singer/concert_singer.sqlite"


def test_returns_resultset_with_columns_and_rows():
    result = execute_sql("SELECT Name FROM singer LIMIT 3", DB_PATH)
    assert isinstance(result, ResultSet)
    assert result.columns == ["Name"]
    assert len(result.rows) == 3
    assert "Name" in result.rows[0]


def test_returns_error_on_bad_sql():
    result = execute_sql("SELECT * FROM no_such_table", DB_PATH)
    assert isinstance(result, Error)
    assert result.kind == "sqlite_error"


def test_max_rows_caps_the_result():
    result = execute_sql("SELECT * FROM singer", DB_PATH, max_rows=2)
    assert isinstance(result, ResultSet)
    assert len(result.rows) == 2


def test_no_max_rows_returns_everything():
    result = execute_sql("SELECT * FROM singer", DB_PATH)
    assert isinstance(result, ResultSet)
    assert len(result.rows) == 6  # all singers in concert_singer


def test_timeout_interrupts_runaway_query():
    sql = (
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) "
        "SELECT count(*) FROM r"
    )
    result = execute_sql(sql, DB_PATH, timeout=0.1)
    assert isinstance(result, Error)
    assert result.kind == "timeout"


def test_non_select_returns_empty_resultset():
    result = execute_sql("BEGIN", DB_PATH)  # no result set, and permitted read-only
    assert isinstance(result, ResultSet)
    assert result.columns == []
    assert result.rows == []
