"""Extract and render low-cardinality column values for value linking."""

import sqlite3

from agent.schema import extract_schema

_TEXT_HINTS = ("char", "text", "clob")


def extract_categorical_values(
    db_path: str, *, max_distinct: int = 100
) -> dict[str, list[str]]:
    """Map "table.column" -> its distinct values, for low-cardinality text columns."""
    schema = extract_schema(db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        out: dict[str, list[str]] = {}
        for table in schema.tables:
            keys = set(table.primary_key) | {fk.column for fk in table.foreign_keys}
            for col in table.columns:
                if col.name in keys or not _is_text(col.type):
                    continue
                cur.execute(f'SELECT COUNT(DISTINCT "{col.name}") FROM "{table.name}"')
                (n,) = cur.fetchone()
                if not 0 < n <= max_distinct:
                    continue
                cur.execute(
                    f'SELECT DISTINCT "{col.name}" FROM "{table.name}" '
                    f'WHERE "{col.name}" IS NOT NULL'
                )
                out[f"{table.name}.{col.name}"] = [str(v) for (v,) in cur.fetchall()]
        return out
    finally:
        conn.close()


def _is_text(sql_type: str) -> bool:
    return any(hint in sql_type.lower() for hint in _TEXT_HINTS)
