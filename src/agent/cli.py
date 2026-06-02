import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import anthropic
from anthropic.types import TextBlock, ToolParam, ToolUseBlock

from agent.executor import Error, execute_sql
from agent.messages import MessageManager
from agent.prompt import build_schema_prompt
from agent.retrieval import CHROMA_PATH, retrieve

EXECUTE_SQL_TOOL: ToolParam = {
    "name": "execute_sql",
    "description": "Run a read-only SQL query and return the resulting rows.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A single read-only SELECT statement.",
            }
        },
        "required": ["query"],
    },
}

MODEL = "claude-sonnet-4-6"

SYSTEM = (
    "You are a text-to-SQL assistant. "
    "Use the execute_sql tool to run a query against the database, "
    "then answer the question from the results. "
    "Use ONLY tables and columns that exist in the schema. "
    "Select exactly the columns the question asks for and no others — "
    "do not add columns you only used for sorting, grouping, or filtering "
    "(for example, a COUNT you ORDER BY) unless the question asks for them.\n\n"
    "Schema:\n{schema}"
)


def _build_manager(
    question: str, db_path: str, *, chroma_path: str = CHROMA_PATH
) -> MessageManager:
    tables = retrieve(question, k=3, chroma_path=chroma_path)
    schema = build_schema_prompt(tables, db_path)
    manager = MessageManager(system=SYSTEM.format(schema=schema))
    manager.add_user_message(question)
    return manager


def generate_sql(
    question: str, db_path: str, *, chroma_path: str = CHROMA_PATH
) -> str | None:
    """Run the RAG NL-to-SQL step and return the model's query (no execution)."""
    manager = _build_manager(question, db_path, chroma_path=chroma_path)
    response = anthropic.Anthropic().messages.create(
        model=MODEL,
        max_tokens=2048,
        tools=[EXECUTE_SQL_TOOL],
        **manager.to_anthropic_kwargs(),
    )
    tool_use = next((b for b in response.content if isinstance(b, ToolUseBlock)), None)
    return cast(str, tool_use.input["query"]) if tool_use else None


def answer_question(question: str, db_path: str) -> str:
    manager = _build_manager(question, db_path)

    client = anthropic.Anthropic()

    # First call: the model reasons, then requests the execute_sql tool.
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        tools=[EXECUTE_SQL_TOOL],
        **manager.to_anthropic_kwargs(),
    )
    manager.add_assistant_message([block.model_dump() for block in response.content])

    tool_use = next((b for b in response.content if isinstance(b, ToolUseBlock)), None)
    if tool_use is None:
        # Model answered without querying; return whatever text it produced.
        return next((b.text for b in response.content if isinstance(b, TextBlock)), "")

    # Run the requested tool, feed the result back as a tool_result block.
    query = cast(str, tool_use.input["query"])
    result = execute_sql(query, db_path, max_rows=100)
    if isinstance(result, Error):
        _log_sql_error(question, query, result.message)
        content = f"Error: {result.message}"
    else:
        content = json.dumps(result.rows, default=str)
    manager.add_user_message(
        [
            {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": content,
            }
        ]
    )

    # Second call: the model turns the rows into a natural-language answer.
    final = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        tools=[EXECUTE_SQL_TOOL],
        **manager.to_anthropic_kwargs(),
    )
    return next((b.text for b in final.content if isinstance(b, TextBlock)), "")


def _log_sql_error(question: str, sql: str, error: str) -> None:
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "question": question,
        "sql": sql,
        "error": error,
    }
    Path("logs").mkdir(exist_ok=True)
    with open("logs/sql_errors.jsonl", "a") as log:
        log.write(json.dumps(record) + "\n")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python -m agent.cli '<question>' <db_path>")
        sys.exit(1)

    question, db_path = sys.argv[1], sys.argv[2]
    print(answer_question(question, db_path))


if __name__ == "__main__":
    main()
