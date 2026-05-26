"""Build the schema section for the prompt from retrieved tables."""

from agent.schema import Table, read_schema_map
from agent.values import extract_categorical_values


def build_schema_prompt(tables: list[Table], db_path: str) -> str:
    ddl = read_schema_map(db_path)
    values = extract_categorical_values(db_path)
    blocks = []
    for table in tables:
        block = ddl[table.name]
        samples = [
            f"{key.split('.', 1)[1]}: {', '.join(vals)}"
            for key, vals in values.items()
            if key.startswith(f"{table.name}.")
        ]
        if samples:
            block += "\n-- sample values: " + "; ".join(samples)
        blocks.append(block)
    return "\n\n".join(blocks)
