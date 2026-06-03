"""Run read-only SQL; return a typed result set or a structured error."""

import sqlite3
import threading
from dataclasses import dataclass

import sqlglot
from sqlglot import expressions as exp


@dataclass
class ResultSet:
    columns: list[str]
    rows: list[dict]


@dataclass
class Error:
    kind: str
    message: str


def execute_sql(
    sql: str,
    db_path: str,
    *,
    timeout: float = 5.0,
    max_rows: int | None = None,
) -> ResultSet | Error:
    """Run a query read-only; return rows or a structured error."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    timed_out = False

    def fire():
        nonlocal timed_out
        timed_out = True
        conn.interrupt()

    timer = threading.Timer(timeout, fire)
    timer.start()
    try:
        cur = conn.execute(sql)
        if cur.description is None:
            return ResultSet(columns=[], rows=[])
        columns = [c[0] for c in cur.description]
        raw = cur.fetchall() if max_rows is None else cur.fetchmany(max_rows)
        rows = [dict(zip(columns, row, strict=True)) for row in raw]
        return ResultSet(columns=columns, rows=rows)
    except sqlite3.Error as exc:
        return Error(kind="timeout" if timed_out else "sqlite_error", message=str(exc))
    finally:
        timer.cancel()
        conn.close()


_READ_ONLY = (exp.Select, exp.Union, exp.Intersect, exp.Except)


def safe_execute(sql: str, db_path: str, *, row_limit: int = 100) -> ResultSet | Error:
    """Reject non-read-only SQL, cap rows, then execute. For the runtime path."""
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.ParseError as exc:
        return Error(kind="rejected", message=f"unparseable SQL: {exc}")
    if not isinstance(parsed, _READ_ONLY):
        return Error(kind="rejected", message=f"{type(parsed).__name__} not read-only")
    if isinstance(parsed, exp.Select) and parsed.args.get("limit") is None:
        parsed = parsed.limit(row_limit)
    return execute_sql(parsed.sql(dialect="sqlite"), db_path, max_rows=row_limit)
