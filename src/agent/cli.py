import sys
from typing import cast

import anthropic
from anthropic.types import TextBlock, ToolParam, ToolUseBlock

from agent.executor import execute_sql
from agent.messages import MessageManager
from agent.retrieval import top_k_similar
from agent.schema import read_schema_map

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

TABLE_BLURBS = {
    "singer": "singer: recording artists — their country, age, and song release year.",
    "concert": "concert: concert events — their theme, year, and host stadium.",
    "stadium": "stadium: venues — their location and seating-capacity statistics.",
    "singer_in_concert": "singer_in_concert: junction table linking singers to the concerts they performed in.",
}


def answer_question(question: str, db_path: str) -> str:
    blurb_to_table = {blurb: table for table, blurb in TABLE_BLURBS.items()}
    blurbs = top_k_similar(question, list(blurb_to_table),k=3)
    schema_map = read_schema_map(db_path)
    schema = "\n\n".join(schema_map[blurb_to_table[b]] for b in blurbs)
    blurb_context = "\n".join(blurbs)    

    system = (
        "You are a text-to-SQL assistant. "
        "Use the execute_sql tool to run a query against the database, "
        "then answer the question from the results. "
        "Use ONLY tables and columns that exist in the schema.\n\n"
        f"{blurb_context}\n\nSchema:\n{schema}"
    )

    manager = MessageManager(system=system)
    manager.add_user_message(question)

    client = anthropic.Anthropic()

    # First call: the model reasons, then requests the execute_sql tool.
    response = client.messages.create(
        model="claude-sonnet-4-6",
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
    result = execute_sql(query, db_path)
    manager.add_user_message(
        [
            {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": str(result),
            }
        ]
    )

    # Second call: the model turns the rows into a natural-language answer.
    final = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        tools=[EXECUTE_SQL_TOOL],
        **manager.to_anthropic_kwargs(),
    )
    return next((b.text for b in final.content if isinstance(b, TextBlock)), "")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m agent.cli '<your question>'")
        sys.exit(1)

    question = sys.argv[1]
    db_path = "sqlite-data/concert_singer/concert_singer.sqlite"

    print(answer_question(question, db_path))


if __name__ == "__main__":
    main()
