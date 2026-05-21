"""Format a SQLite schema for inclusion in LLM prompts."""

import sqlite3


def read_schema(db_path: str) -> str:
    """Return all CREATE TABLE statements joined by blank lines."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT sql FROM sqlite_schema "
            "WHERE type='table' AND sql IS NOT NULL "
            "ORDER BY name"
        )
        statements = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()
    return "\n\n".join(statements)
