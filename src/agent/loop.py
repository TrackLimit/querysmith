"""Self-correcting agent loop: generate -> execute -> fix, bounded to a few retries."""

from agent.cli import ask_for_sql, generate_sql
from agent.executor import Error, ResultSet, execute_sql
from agent.messages import MessageManager
from agent.prompt import build_schema_prompt
from agent.retrieval import CHROMA_PATH, retrieve

MAX_RETRIES = 3


def _is_retryable(error: Error) -> bool:
    # syntax / wrong-column errors are fixable by re-prompting; a timed-out runaway
    # query won't improve on retry.
    return error.kind == "sqlite_error"


def _correction_prompt(attempts: list[tuple[str, str]], schema: str) -> str:
    history = "\n".join(
        f"{i}. {sql}\n   error: {err}" for i, (sql, err) in enumerate(attempts, 1)
    )
    return (
        "You are fixing a SQL query that failed. Use the execute_sql tool to run a "
        "corrected, read-only query. Use ONLY tables and columns in the schema, and "
        "do not repeat any failed attempt below.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Failed attempts:\n{history}"
    )


def _regenerate(
    question: str, db_path: str, attempts: list[tuple[str, str]], *, chroma_path: str
) -> str | None:
    tables = retrieve(f"{question} {attempts[-1][1]}", k=3, chroma_path=chroma_path)
    schema = build_schema_prompt(tables, db_path)
    manager = MessageManager(system=_correction_prompt(attempts, schema))
    manager.add_user_message(question)
    return ask_for_sql(manager)


def solve_sql(
    question: str, db_path: str, *, chroma_path: str = CHROMA_PATH
) -> str | None:
    """Generate SQL, then self-correct on execution errors (up to MAX_RETRIES)."""
    sql = generate_sql(question, db_path, chroma_path=chroma_path)
    attempts: list[tuple[str, str]] = []
    for _ in range(MAX_RETRIES):
        if sql is None:
            return None
        result = execute_sql(sql, db_path)
        if isinstance(result, ResultSet) or not _is_retryable(result):
            return sql  # ran clean, or unrecoverable (timeout) — hand back the last SQL
        attempts.append((sql, result.message))
        sql = _regenerate(question, db_path, attempts, chroma_path=chroma_path)
    return sql
