"""Run read-only SQL against a SQLite database."""

import sqlite3


def execute_sql(sql: str, db_path: str, max_rows: int = 1000) -> list[dict] | str:
    """Run a query read-only; return rows (capped at max_rows) or the error string."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        if cur.description is None:  # non-SELECT statements have no result set
            return []
        columns = [col[0] for col in cur.description]
        rows = cur.fetchmany(max_rows)
        return [dict(zip(columns, row, strict=True)) for row in rows]
    except sqlite3.Error as exc:
        return f"SQL error: {exc}"
    finally:
        conn.close()
