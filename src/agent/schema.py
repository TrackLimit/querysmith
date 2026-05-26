"""Extract a SQLite schema into typed objects and render it for embedding/prompts."""

import sqlite3

from pydantic import BaseModel, Field


class Column(BaseModel):
    name: str
    type: str  # declared type, verbatim; may be empty ("INT", "VARCHAR(10)", "")


class ForeignKey(BaseModel):
    column: str
    ref_table: str
    ref_column: str


class Table(BaseModel):
    name: str
    columns: list[Column]
    primary_key: list[str]  # column names; a list so composite PKs fit
    foreign_keys: list[ForeignKey]
    sample_values: dict[str, list[str]] = Field(default_factory=dict)


class Schema(BaseModel):
    tables: list[Table]


def extract_schema(db_path: str) -> Schema:
    """Read every user table's columns, primary key, and foreign keys."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_schema "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        names = [row[0] for row in cur.fetchall()]
        return Schema(tables=[_read_table(cur, name) for name in names])
    finally:
        conn.close()


def _read_table(cur: sqlite3.Cursor, name: str) -> Table:
    cur.execute("SELECT name, type, pk FROM pragma_table_info(?)", (name,))
    columns: list[Column] = []
    pk: list[tuple[int, str]] = []
    for col_name, col_type, pk_pos in cur.fetchall():
        columns.append(Column(name=col_name, type=col_type))
        if pk_pos:  # 0 = not part of the PK; 1, 2, ... = position within it
            pk.append((pk_pos, col_name))
    primary_key = [col for _, col in sorted(pk)]

    cur.execute('SELECT "from", "table", "to" FROM pragma_foreign_key_list(?)', (name,))
    foreign_keys = [
        ForeignKey(column=frm, ref_table=ref_table, ref_column=ref_col)
        for frm, ref_table, ref_col in cur.fetchall()
    ]
    return Table(
        name=name, columns=columns, primary_key=primary_key, foreign_keys=foreign_keys
    )


def render_table(table: Table) -> str:
    """One self-contained sentence-form description, for embedding."""
    pk = set(table.primary_key)
    cols = []
    for col in table.columns:
        label = col.type.lower() or "unknown"
        if col.name in pk:
            label += ", primary key"
        cols.append(f"{col.name} ({label})")
    text = f"Table {table.name}. Columns: {', '.join(cols)}."

    if table.foreign_keys:
        fks = "; ".join(
            f"{fk.column} references {fk.ref_table}({fk.ref_column})"
            for fk in table.foreign_keys
        )
        text += f" Foreign keys: {fks}."
    else:
        text += " No foreign keys."
    return text


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
