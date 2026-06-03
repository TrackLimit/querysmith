from agent.executor import Error, ResultSet, safe_execute

DB = "sqlite-data/concert_singer/concert_singer.sqlite"


def test_select_is_allowed():
    assert isinstance(safe_execute("SELECT Name FROM singer", DB), ResultSet)


def test_cte_is_allowed():
    sql = "WITH s AS (SELECT * FROM singer) SELECT Name FROM s"
    assert isinstance(safe_execute(sql, DB), ResultSet)


def test_non_readonly_statements_are_refused():
    for sql in (
        "DROP TABLE singer",
        "DELETE FROM singer",
        "INSERT INTO singer (Name) VALUES ('x')",
        "UPDATE singer SET Name = 'x'",
        "ATTACH DATABASE 'evil.db' AS e",
    ):
        result = safe_execute(sql, DB)
        assert isinstance(result, Error) and result.kind == "rejected"


def test_row_cap_is_injected():
    result = safe_execute("SELECT * FROM singer", DB, row_limit=2)
    assert isinstance(result, ResultSet) and len(result.rows) == 2


def test_union_capped_via_max_rows():
    sql = "SELECT Name FROM singer UNION SELECT Country FROM singer"
    result = safe_execute(sql, DB, row_limit=2)
    assert isinstance(result, ResultSet) and len(result.rows) == 2
