"""Format a SQLite schema for inclusion in LLM prompts."""

import sqlite3


def read_schema_map(db_path: str) -> dict[str, str]:
    """Return {table_name: CREATE statement} for every table."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, sql FROM sqlite_schema "
            "WHERE type='table' AND sql IS NOT NULL "
            "ORDER BY name"
        )
        return {name: sql for name, sql in cur.fetchall()}
    finally:
        conn.close()


def read_schema(db_path: str) -> str:
    """Return all CREATE TABLE statements joined by blank lines."""
    return "\n\n".join(read_schema_map(db_path).values())
