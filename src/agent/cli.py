import sys

import anthropic
from pydantic import BaseModel

from agent.messages import MessageManager
from agent.schema import read_schema


class SqlDraft(BaseModel):
    reasoning: str
    sql: str


def draft_sql(question: str, db_path: str) -> SqlDraft:
    schema = read_schema(db_path)

    system = (
        "You are a text-to-SQL assistant. "
        "Given a SQLite schema and a natural-language question, "
        "generate a SQL query that answers it. "
        "Use ONLY tables and columns that exist in the schema.\n\n"
        f"Schema:\n{schema}"
    )

    manager = MessageManager(system=system)
    manager.add_user_message(question)

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        output_format=SqlDraft,
        **manager.to_anthropic_kwargs(),
    )

    return response.parsed_output


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m agent.cli '<your question>'")
        sys.exit(1)

    question = sys.argv[1]
    db_path = "sqlite-data/concert_singer/concert_singer.sqlite"

    draft = draft_sql(question, db_path)
    print(f"Reasoning:\n{draft.reasoning}\n")
    print(f"SQL:\n{draft.sql}\n")


if __name__ == "__main__":
    main()
